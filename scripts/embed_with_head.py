import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import csv

# ------------------ utils ------------------

def letterbox_to_square(pil_img: Image.Image, size: int, pad_value: int = 128) -> Image.Image:
    w, h = pil_img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = pil_img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp','.JPG','.JPEG','.PNG','.BMP','.TIF','.TIFF','.WEBP'}

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix in IMG_EXTS])

# ------------------ model wrappers ------------------

class DinoBackbone(nn.Module):
    def __init__(self, model_name: str, device: str):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

class EmbedderWithHead(nn.Module):
    def __init__(self, ckpt_path: Path, device: str):
        super().__init__()
        obj = torch.load(ckpt_path, map_location='cpu')
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.backbone = DinoBackbone(obj['dino_model'], device)
        # infer feat dim from backbone output
        with torch.no_grad():
            dummy = torch.zeros(1,3,obj['image_size'],obj['image_size'])
            feat_dim = self.backbone(dummy).shape[1]
        self.head = nn.Sequential(nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, obj['embed_dim'], bias=False))
        self.head.load_state_dict(obj['state_dict_head'])
        self.to(self.device)
        self.size = obj['image_size']
        self.tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox_to_square(im, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def embed_image(self, pil_img: Image.Image) -> torch.Tensor:
        x = self.tf(pil_img)
        feats = self.backbone(x.unsqueeze(0).to(self.device))
        emb = F.normalize(self.head(feats), dim=-1)
        return emb.squeeze(0).cpu()

# ------------------ matching ------------------

@torch.no_grad()
def knn_best_per_class(emb: torch.Tensor, gallery_embs: torch.Tensor, labels: List[str], topk: int = 5):
    sims = torch.mv(gallery_embs, emb)  # (N,)
    M = min(200, sims.numel())
    vals, idxs = torch.topk(sims, k=M)
    best: Dict[str, float] = {}
    for v, i in zip(vals.tolist(), idxs.tolist()):
        c = labels[i]
        if c not in best or v > best[c]:
            best[c] = v
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return [(c, float(s)) for c, s in ranked]

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description='Export gallery/probe embeddings with trained head (k-NN ready)')
    ap.add_argument('--embedder', type=Path, required=True, help='checkpoint from train_arcface_head.py')
    ap.add_argument('--gallery', type=Path, required=True, help='folder with class subfolders (e.g., out/cv/fold_xx/train)')
    ap.add_argument('--probe', type=Path, default=None, help='optional probe folder (e.g., out/cv/fold_xx/val) to evaluate')
    ap.add_argument('--save_index', type=Path, required=True)
    ap.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()

    emb = EmbedderWithHead(args.embedder, args.device)

    # Build gallery
    classes = sorted([d for d in args.gallery.iterdir() if d.is_dir() and not d.name.startswith('.')], key=lambda p: p.name.lower())
    g_embs, g_labels = [], []
    for cls_dir in classes:
        for img_path in list_images(cls_dir):
            e = emb.embed_image(Image.open(img_path).convert('RGB'))
            g_embs.append(e)
            g_labels.append(cls_dir.name)
    if len(g_embs) == 0:
        raise SystemExit('No gallery images found.')
    G = F.normalize(torch.stack(g_embs, dim=0), dim=-1)

    # Save index
    args.save_index.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'embeddings': G.cpu(), 'labels': g_labels}, args.save_index)
    print(f"Saved index: {args.save_index.resolve()}  (embeddings: {G.shape})")

    # Optional probe evaluation
    if args.probe is not None:
        total = 0
        top1 = 0
        topk = 0
        out_csv = Path('out/eval_results_head.csv')
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image','true_whale','predicted','score','topk_labels'])
            for cls_dir in sorted([d for d in args.probe.iterdir() if d.is_dir() and not d.name.startswith('.')], key=lambda p: p.name.lower()):
                for img_path in list_images(cls_dir):
                    e = emb.embed_image(Image.open(img_path).convert('RGB'))
                    preds = knn_best_per_class(e, G, g_labels, topk=args.topk)
                    pred_cls, score = preds[0]
                    labs = [c for c,_ in preds]
                    total += 1
                    if pred_cls == cls_dir.name:
                        top1 += 1
                    if cls_dir.name in labs:
                        topk += 1
                    writer.writerow([str(img_path), cls_dir.name, pred_cls, score, ';'.join(labs)])
        print(f"Probe Top-1: {top1/max(1,total):.4f}  |  Top-{args.topk}: {topk/max(1,total):.4f}")

if __name__ == '__main__':
    main()
