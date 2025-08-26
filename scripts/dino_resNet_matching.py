import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import csv
from collections import defaultdict, Counter
import numpy as np
import cv2

# ==============================
# Utilities & dataset
# ==============================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image(p)])

class SimpleImageFolder:
    """Minimal folder dataset: expects subfolders per class (whale ID)."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.samples: List[Tuple[Path, str]] = []
        for cls_dir in sorted([d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith('.')]):
            for img in list_images(cls_dir):
                self.samples.append((img, cls_dir.name))
    def __len__(self):
        return len(self.samples)
    def __iter__(self):
        return iter(self.samples)

# ==============================
# Image preprocessing helpers
# ==============================

def letterbox_to_square(pil_img: Image.Image, size: int, pad_value: int = 128) -> Image.Image:
    """Resize image by longer side and pad to a square canvas of given size."""
    w, h = pil_img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = pil_img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    # center paste
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas

def make_edge_image(pil_img: Image.Image, size: int, square_mode: str = "pad") -> Image.Image:
    """Convert to edge-only 3-channel PIL image suitable for ImageNet-pretrained nets.
    Steps: grayscale -> CLAHE -> GaussianBlur -> Canny -> dilate -> invert -> letterbox/center-crop.
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # auto thresholds from median
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    # thicken slightly to reduce fragmentation
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    # invert so fin edges are bright
    edge_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)
    edge_pil = Image.fromarray(edge_rgb)
    if square_mode == "pad":
        edge_pil = letterbox_to_square(edge_pil, size)
    else:
        edge_pil = edge_pil.resize((size, size), Image.BICUBIC)
    return edge_pil

# ==============================
# Backbones (frozen): DINOv2 + ResNet50
# ==============================
class DinoEmbedder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 518, device: str = "cuda",
                 input_mode: str = "rgb", square_mode: str = "center_crop"):
        super().__init__()
        # Optional project-local cache
        self._patch_torch_home()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)
        self.input_mode = input_mode
        self.square_mode = square_mode
        self.image_size = image_size
        # Base normalization for ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if square_mode == "pad":
            self.preproc_rgb = transforms.Compose([
                transforms.Lambda(lambda im: letterbox_to_square(im, image_size)),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.preproc_rgb = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ])
        self.preproc_edge = transforms.Compose([
            transforms.Lambda(lambda im: make_edge_image(im, image_size, square_mode=self.square_mode)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _patch_torch_home(self):
        # allow env var TORCH_HOME to be set by caller; nothing to do here
        return

    @torch.no_grad()
    def _forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model(batch.to(self.device))  # (B, D)
        feats = F.normalize(feats, dim=-1)
        return feats

    def preproc_preview(self, img: Image.Image) -> Image.Image:
        # Return PIL image after our geometric transform (for dumping)
        if self.input_mode == "edge":
            return make_edge_image(img, self.image_size, square_mode=self.square_mode)
        return letterbox_to_square(img, self.image_size) if self.square_mode == "pad" else img.resize((self.image_size, self.image_size), Image.BICUBIC)

    def embed_image(self, img: Image.Image, tta: int = 4, hflip: bool = True) -> torch.Tensor:
        views = []
        base = (self.preproc_edge if self.input_mode == "edge" else self.preproc_rgb)(img)
        views.append(base)
        if hflip:
            views.append(torch.flip(base, dims=[2]))
        if tta > 2:
            scale_aug = transforms.RandomResizedCrop(size=base.shape[1], scale=(0.85, 1.0))
            for _ in range(tta - 2):
                views.append(scale_aug(img))
        batch = torch.stack(views, dim=0)
        embs = self._forward_batch(batch)
        emb = F.normalize(embs.mean(dim=0), dim=-1)
        return emb.cpu()

class ResNet50Embedder(nn.Module):
    def __init__(self, image_size: int = 448, device: str = "cuda", input_mode: str = "rgb", square_mode: str = "center_crop"):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        except AttributeError:
            weights = None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)
        self.input_mode = input_mode
        self.square_mode = square_mode
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if square_mode == "pad":
            self.preproc_rgb = transforms.Compose([
                transforms.Lambda(lambda im: letterbox_to_square(im, image_size)),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.preproc_rgb = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ])
        self.preproc_edge = transforms.Compose([
            transforms.Lambda(lambda im: make_edge_image(im, image_size, square_mode=self.square_mode)),
            transforms.ToTensor(),
            self.normalize,
        ])

    @torch.no_grad()
    def _forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        feats = self.model(batch.to(self.device))  # (B, 2048)
        feats = F.normalize(feats, dim=-1)
        return feats

    def preproc_preview(self, img: Image.Image) -> Image.Image:
        if self.input_mode == "edge":
            return make_edge_image(img, self.image_size, square_mode=self.square_mode)
        return letterbox_to_square(img, self.image_size) if self.square_mode == "pad" else img.resize((self.image_size, self.image_size), Image.BICUBIC)

    def embed_image(self, img: Image.Image, tta: int = 4, hflip: bool = True) -> torch.Tensor:
        views = []
        base = (self.preproc_edge if self.input_mode == "edge" else self.preproc_rgb)(img)
        views.append(base)
        if hflip:
            views.append(torch.flip(base, dims=[2]))
        if tta > 2:
            scale_aug = transforms.RandomResizedCrop(size=base.shape[1], scale=(0.85, 1.0))
            for _ in range(tta - 2):
                views.append(scale_aug(img))
        batch = torch.stack(views, dim=0)
        embs = self._forward_batch(batch)
        emb = F.normalize(embs.mean(dim=0), dim=-1)
        return emb.cpu()

# ==============================
# Gallery embedding & modes
# ==============================

def embed_gallery(embedder, gallery_root: Path, tta: int = 1, hflip: bool = True,
                  dump_dir: Optional[Path] = None, dump_limit: int = 0):
    embs: List[torch.Tensor] = []
    labels: List[str] = []
    ds = SimpleImageFolder(gallery_root)
    dumped = 0
    for img_path, cls in ds:
        img = Image.open(img_path).convert("RGB")
        if dump_dir and dumped < dump_limit:
            dump_dir.mkdir(parents=True, exist_ok=True)
            prev = embedder.preproc_preview(img)
            prev.save(dump_dir / f"preview_{dumped:05d}_{cls}.jpg")
            dumped += 1
        emb = embedder.embed_image(img, tta=tta, hflip=hflip)
        embs.append(emb)
        labels.append(cls)
    if len(embs) == 0:
        return torch.empty(0, 1), []
    return torch.stack(embs, dim=0), labels  # (N, D), [N]

def build_prototypes_from_embs(embs: torch.Tensor, labels: List[str], mode: str = "prototype", trim: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    mode: 'prototype' -> simple mean
          'robust'    -> trimmed mean (drop farthest trim fraction) or medoid fallback
    """
    by_cls: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        by_cls[c].append(i)
    out: Dict[str, torch.Tensor] = {}
    for c, idxs in by_cls.items():
        M = embs[idxs]  # (m, D)
        if mode == "prototype" or len(idxs) <= 2:
            proto = F.normalize(M.mean(dim=0), dim=-1)
        else:
            # robust: drop farthest trim fraction by cosine distance to mean
            mean0 = M.mean(dim=0, keepdim=True)  # (1, D)
            mean0 = F.normalize(mean0, dim=-1)
            sims = (M @ mean0.T).squeeze(1)  # cosine similarity
            k_keep = max(1, int(round((1.0 - trim) * M.shape[0])))
            keep_idx = torch.topk(sims, k=k_keep, largest=True).indices
            proto = F.normalize(M[keep_idx].mean(dim=0), dim=-1)
        out[c] = proto.cpu()
    return out

@torch.no_grad()
def match_probe_prototypes(emb: torch.Tensor, prototypes: Dict[str, torch.Tensor], topk: int = 5):
    cls_names = list(prototypes.keys())
    protos = torch.stack([prototypes[c] for c in cls_names], dim=0)  # (C, D)
    sims = torch.mv(protos, emb)  # cosine similarity (L2-normalized)
    vals, idx = torch.topk(sims, k=min(topk, sims.numel()))
    return [(cls_names[i], float(vals[j])) for j, i in enumerate(idx.tolist())]

@torch.no_grad()
def match_probe_knn(emb: torch.Tensor, gallery_embs: torch.Tensor, labels: List[str], topk: int = 5):
    """Aggregate by class using the **maximum cosine** over that class's gallery images."""
    if gallery_embs.numel() == 0:
        return []
    sims = torch.mv(gallery_embs, emb)  # (N,)
    # For speed: take top M global, then reduce; M heuristic = top 200
    M = min(200, sims.numel())
    vals, idxs = torch.topk(sims, k=M)
    best_by_class: Dict[str, float] = {}
    for v, i in zip(vals.tolist(), idxs.tolist()):
        c = labels[i]
        if c not in best_by_class or v > best_by_class[c]:
            best_by_class[c] = v
    # rank classes by best score
    ranked = sorted(best_by_class.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return [(c, float(s)) for c, s in ranked]

@torch.no_grad()
def match_fused(emb_d: torch.Tensor, emb_r: torch.Tensor,
               mode: str,
               prot_d: Optional[Dict[str, torch.Tensor]] = None,
               prot_r: Optional[Dict[str, torch.Tensor]] = None,
               embs_d: Optional[torch.Tensor] = None,
               labels_d: Optional[List[str]] = None,
               embs_r: Optional[torch.Tensor] = None,
               labels_r: Optional[List[str]] = None,
               alpha: float = 0.6, topk: int = 5):
    """Fuse class scores from two backbones. mode in {'prototype','robust','knn'}.
    For 'prototype'/'robust' we require prot_* dicts; for 'knn' we require embs_* and labels_*.
    """
    # Build per-class score dicts
    scores: Dict[str, float] = defaultdict(float)
    if mode in {"prototype", "robust"}:
        # Align class lists
        classes = sorted(set(prot_d.keys()) & set(prot_r.keys()))
        for c in classes:
            sd = float(torch.dot(prot_d[c], emb_d)) if prot_d else -1e9
            sr = float(torch.dot(prot_r[c], emb_r)) if prot_r else -1e9
            scores[c] = alpha * sd + (1 - alpha) * sr
    else:  # knn
        # compute top candidates from each branch then combine by weighted sum
        top_d = match_probe_knn(emb_d, embs_d, labels_d, topk=max(200, topk)) if embs_d is not None else []
        top_r = match_probe_knn(emb_r, embs_r, labels_r, topk=max(200, topk)) if embs_r is not None else []
        for c, s in top_d:
            scores[c] = max(scores[c], alpha * s)
        for c, s in top_r:
            scores[c] = max(scores[c], scores.get(c, -1e9))  # ensure exists
            scores[c] = max(scores[c], (1 - alpha) * s) if c in scores else (1 - alpha) * s
        # If a class appears in only one backbone's shortlist, keep that score
        for c, s in top_r:
            if c not in scores:
                scores[c] = (1 - alpha) * s
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return [(c, float(s)) for c, s in ranked]

# ==============================
# Save/load gallery index
# ==============================

def save_index(path: Path,
               dino: Optional[dict] = None,
               res: Optional[dict] = None,
               meta: Optional[dict] = None):
    payload = {"dinov2": dino, "resnet50": res, "meta": meta or {}}
    # Tensors must be on CPU
    def to_cpu(obj):
        if obj is None:
            return None
        out = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
            elif torch.is_tensor(v):
                out[k] = v.cpu()
            else:
                out[k] = v
        return out
    payload["dinov2"] = to_cpu(dino)
    payload["resnet50"] = to_cpu(res)
    torch.save(payload, path)


def load_index(path: Path):
    obj = torch.load(path, map_location="cpu")
    # Backward-compat: if file contained only prototypes
    if "dinov2" not in obj and "resnet50" not in obj:
        # Assume obj is {class: tensor}
        return {"dinov2": {"type": "prototype", "prototypes": obj}, "resnet50": None, "meta": {}}
    # normalize tensors
    for k in ["dinov2", "resnet50"]:
        if obj.get(k) and obj[k].get("prototypes"):
            for c, v in obj[k]["prototypes"].items():
                obj[k]["prototypes"][c] = F.normalize(v, dim=-1)
        if obj.get(k) and obj[k].get("embeddings") is not None:
            obj[k]["embeddings"] = F.normalize(obj[k]["embeddings"], dim=-1)
    return obj

# ==============================
# Evaluation
# ==============================

def evaluate(backbone: str,
             gallery_root: Path,
             probe_root: Path,
             dino_model: str,
             dino_size: int,
             res_size: int,
             device: str,
             tta: int,
             hflip: bool,
             out_csv: Path,
             topk: int = 5,
             alpha: float = 0.6,
             input_mode: str = "rgb",
             square_mode: str = "center_crop",
             gallery_mode: str = "prototype",
             robust_trim: float = 0.2,
             load_index_path: Optional[Path] = None,
             dump_preproc: int = 0,
             limit_probe: Optional[int] = None):

    # Try to load existing gallery index
    cached = None
    if load_index_path and load_index_path.exists():
        cached = load_index(load_index_path)

    # Build/prepare gallery for DINO
    dino = None
    dino_pack = None
    if backbone in {"dinov2", "both"}:
        dino = DinoEmbedder(model_name=dino_model, image_size=dino_size, device=device,
                            input_mode=input_mode, square_mode=square_mode)
        if cached and cached.get("dinov2"):
            dino_pack = cached["dinov2"]
        else:
            g_embs_d, g_labels_d = embed_gallery(dino, gallery_root, tta=tta, hflip=hflip,
                                                 dump_dir=Path("out/preproc_dump_dino") if dump_preproc > 0 else None,
                                                 dump_limit=dump_preproc)
            if gallery_mode in {"prototype", "robust"}:
                prot_d = build_prototypes_from_embs(g_embs_d, g_labels_d, mode=gallery_mode, trim=robust_trim)
                dino_pack = {"type": gallery_mode, "prototypes": prot_d, "embeddings": None, "labels": None}
            else:  # knn
                dino_pack = {"type": "knn", "prototypes": None, "embeddings": F.normalize(g_embs_d, dim=-1), "labels": g_labels_d}

    # Build/prepare gallery for ResNet
    res = None
    res_pack = None
    if backbone in {"resnet50", "both"}:
        res = ResNet50Embedder(image_size=res_size, device=device,
                               input_mode=input_mode, square_mode=square_mode)
        if cached and cached.get("resnet50"):
            res_pack = cached["resnet50"]
        else:
            g_embs_r, g_labels_r = embed_gallery(res, gallery_root, tta=tta, hflip=hflip,
                                                 dump_dir=Path("out/preproc_dump_resnet") if dump_preproc > 0 else None,
                                                 dump_limit=dump_preproc)
            if gallery_mode in {"prototype", "robust"}:
                prot_r = build_prototypes_from_embs(g_embs_r, g_labels_r, mode=gallery_mode, trim=robust_trim)
                res_pack = {"type": gallery_mode, "prototypes": prot_r, "embeddings": None, "labels": None}
            else:
                res_pack = {"type": "knn", "prototypes": None, "embeddings": F.normalize(g_embs_r, dim=-1), "labels": g_labels_r}

    # Probe loop
    ds_probe = SimpleImageFolder(probe_root)
    n_total = len(ds_probe)
    if limit_probe is not None:
        n_total = min(n_total, limit_probe)
    top1_correct = 0
    topk_correct = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["image", "true_whale", "predicted_whale", "score", "topk",
                  "backbone", "input_mode", "square_mode", "gallery_mode"]
        writer.writerow(header)

        for i, (img_path, true_cls) in enumerate(ds_probe):
            if i >= n_total:
                break
            img = Image.open(img_path).convert("RGB")

            if backbone == "dinov2":
                emb = dino.embed_image(img, tta=tta, hflip=hflip)
                if dino_pack["type"] in {"prototype", "robust"}:
                    preds = match_probe_prototypes(emb, dino_pack["prototypes"], topk=topk)
                else:
                    preds = match_probe_knn(emb, dino_pack["embeddings"], dino_pack["labels"], topk=topk)
                pred_cls, pred_score = preds[0]

            elif backbone == "resnet50":
                emb = res.embed_image(img, tta=tta, hflip=hflip)
                if res_pack["type"] in {"prototype", "robust"}:
                    preds = match_probe_prototypes(emb, res_pack["prototypes"], topk=topk)
                else:
                    preds = match_probe_knn(emb, res_pack["embeddings"], res_pack["labels"], topk=topk)
                pred_cls, pred_score = preds[0]

            else:  # both
                emb_d = dino.embed_image(img, tta=tta, hflip=hflip)
                emb_r = res.embed_image(img, tta=tta, hflip=hflip)
                preds = match_fused(emb_d, emb_r,
                                    mode=gallery_mode,
                                    prot_d=dino_pack.get("prototypes") if dino_pack else None,
                                    prot_r=res_pack.get("prototypes") if res_pack else None,
                                    embs_d=dino_pack.get("embeddings") if dino_pack else None,
                                    labels_d=dino_pack.get("labels") if dino_pack else None,
                                    embs_r=res_pack.get("embeddings") if res_pack else None,
                                    labels_r=res_pack.get("labels") if res_pack else None,
                                    alpha=alpha, topk=topk)
                pred_cls, pred_score = preds[0]

            topk_labels = [c for c, _ in preds]
            top1_correct += int(pred_cls == true_cls)
            topk_correct += int(true_cls in topk_labels)
            writer.writerow([str(img_path), true_cls, pred_cls, pred_score, ";".join(topk_labels),
                             backbone, input_mode, square_mode, gallery_mode])

    top1 = top1_correct / max(1, n_total)
    topk_acc = topk_correct / max(1, n_total)

    # Prepare packs to save
    packs = {"dinov2": dino_pack, "resnet50": res_pack,
             "meta": {"backbone": backbone, "input_mode": input_mode, "square_mode": square_mode,
                      "gallery_mode": gallery_mode, "dino_model": dino_model}}

    return top1, topk_acc, packs

# ==============================
# CLI
# ==============================

def main():
    parser = argparse.ArgumentParser(description="One-shot dorsal fin matching with DINOv2 + optional ResNet-50 (frozen), with k-NN and caching")
    parser.add_argument("--gallery", type=Path, required=True, help="Path to out/eval/gallery or out/cv/.../train")
    parser.add_argument("--probe", type=Path, required=True, help="Path to out/eval/probe or out/cv/.../val")
    parser.add_argument("--save_index", type=Path, default=Path("out/dino_resnet_index.pt"))
    parser.add_argument("--load_index", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"]) 

    # Performance / env
    parser.add_argument("--threads", type=int, default=None, help="Optional: cap PyTorch CPU threads")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional: set TORCH_HOME to this dir for weights cache")

    # Backbone control
    parser.add_argument("--backbone", type=str, default="dinov2", choices=["dinov2", "resnet50", "both"],
                        help="Use DINOv2 only, ResNet50 only, or fuse both with late-score fusion")

    # DINOv2 options
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14", help="e.g., dinov2_vits14, dinov2_vitb14, dinov2_vitl14")
    parser.add_argument("--dino_image_size", type=int, default=518)

    # ResNet50 options
    parser.add_argument("--resnet_image_size", type=int, default=448)

    # Eval + TTA
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--tta", type=int, default=1, help="#views for test-time augmentation (>=1)")
    parser.add_argument("--no_hflip", action="store_true", help="Disable horizontal flips (useful if left/right fins are distinct)")

    # Fusion
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for DINOv2 in fusion (0..1). 0.6 often works well")

    # Input emphasis
    parser.add_argument("--input_mode", type=str, default="rgb", choices=["rgb", "edge"],
                        help="rgb: original images; edge: Canny edge map to emphasize shape only")
    parser.add_argument("--square_mode", type=str, default="center_crop", choices=["center_crop", "pad"],
                        help="pad: resize long side then pad to square (preserve full fin)")

    # Gallery strategy
    parser.add_argument("--gallery_mode", type=str, default="prototype", choices=["prototype", "robust", "knn"],
                        help="prototype: mean per class; robust: trimmed mean; knn: compare to all gallery images")
    parser.add_argument("--robust_trim", type=float, default=0.2, help="Trim fraction for robust prototype (0..0.5)")

    # Debug / benchmark
    parser.add_argument("--dump_preproc", type=int, default=0, help="Dump N preprocessed gallery images to out/preproc_dump_* for sanity check")
    parser.add_argument("--limit_probe", type=int, default=None, help="Only evaluate the first N probe images (benchmark)")

    args = parser.parse_args()

    # Env / perf tweaks
    if args.cache_dir:
        os.environ["TORCH_HOME"] = str(args.cache_dir.resolve())
    if args.threads:
        try:
            torch.set_num_threads(args.threads)
        except Exception:
            pass

    hflip = not args.no_hflip

    print("[1/3] Building (or loading) gallery index…")
    top1, topk_acc, packs = evaluate(
        backbone=args.backbone,
        gallery_root=args.gallery,
        probe_root=args.probe,
        dino_model=args.dino_model,
        dino_size=args.dino_image_size,
        res_size=args.resnet_image_size,
        device=args.device,
        tta=args.tta,
        hflip=hflip,
        out_csv=Path("out/eval_results.csv"),
        topk=args.topk,
        alpha=args.alpha,
        input_mode=args.input_mode,
        square_mode=args.square_mode,
        gallery_mode=args.gallery_mode,
        robust_trim=args.robust_trim,
        load_index_path=args.load_index,
        dump_preproc=args.dump_preproc,
        limit_probe=args.limit_probe,
    )

    print("[2/3] Saving gallery index…")
    meta = {"backbone": args.backbone, "input_mode": args.input_mode, "square_mode": args.square_mode,
            "gallery_mode": args.gallery_mode, "dino_model": args.dino_model,
            "dino_image_size": args.dino_image_size, "resnet_image_size": args.resnet_image_size}
    save_index(args.save_index,
               dino=packs.get("dinov2"),
               res=packs.get("resnet50"),
               meta=meta)

    print("=== Summary ===")
    print(f"Backbone: {args.backbone}")
    print(f"Input mode: {args.input_mode}  |  Square mode: {args.square_mode}")
    print(f"Gallery mode: {args.gallery_mode}")
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-{args.topk} accuracy: {topk_acc:.4f}")
    print(f"Per-image results: {Path('out/eval_results.csv').resolve()}")
    print(f"Saved gallery index: {args.save_index.resolve()}")

if __name__ == "__main__":
    main()
