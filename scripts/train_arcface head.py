import argparse
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

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

class ListTxtDataset(Dataset):
    def __init__(self, list_path: Path, root: Path, image_size: int, train: bool, no_hflip: bool):
        self.items: List[Tuple[Path, int]] = []
        root = Path(root)
        with open(list_path, 'r') as f:
            for line in f:
                rel, idx = line.strip().split()
                self.items.append((root / rel, int(idx)))
        # transforms
        tfs = [transforms.Lambda(lambda im: letterbox_to_square(im, image_size))]
        if train:
            aug = [
                transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.8),
                transforms.RandomAffine(degrees=6, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=128),
            ]
            if not no_hflip:
                aug.insert(0, transforms.RandomHorizontalFlip(p=0.5))
            tfs.extend(aug)
        tfs.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tf = transforms.Compose(tfs)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert('RGB')
        x = self.tf(img)
        return x, y

# --------------- model (frozen DINO + ArcFace/CosFace head) ---------------

class ArcMarginProduct(nn.Module):
    """ArcFace: cos(θ + m) * s"""
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings)
        W = F.normalize(self.weight)
        cos = F.linear(embeddings, W)  # (N, C)
        idx = torch.arange(cos.size(0), device=cos.device)
        cos_y = cos[idx, labels]
        sin_y = torch.sqrt(torch.clamp(1.0 - cos_y * cos_y, min=1e-7))
        cos_m = cos_y * self.cos_m - sin_y * self.sin_m
        mask = cos_y > self.th
        cos_y_m = torch.where(mask, cos_m, cos_y - self.mm)
        logits = cos.clone()
        logits[idx, labels] = cos_y_m
        logits *= self.s
        return logits

class CosFaceProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings)
        W = F.normalize(self.weight)
        cos = F.linear(embeddings, W)
        idx = torch.arange(cos.size(0), device=cos.device)
        cos[idx, labels] -= self.m
        return cos * self.s

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

class FinIDModel(nn.Module):
    def __init__(self, dino_name: str, embed_dim: int, num_classes: int, margin_type: str, s: float, m: float, device: str):
        super().__init__()
        self.backbone = DinoBackbone(dino_name, device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 518, 518)
            feat_dim = self.backbone(dummy).shape[1]
        self.head = nn.Sequential(nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, embed_dim, bias=False))
        if margin_type == 'arcface':
            self.margin = ArcMarginProduct(embed_dim, num_classes, s=s, m=m)
        else:
            self.margin = CosFaceProduct(embed_dim, num_classes, s=s, m=m)
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.to(self.device)

    def forward(self, x, y=None):
        feats = self.backbone(x)
        emb = F.normalize(self.head(feats), dim=-1)
        if y is None:
            return emb
        logits = self.margin(emb, y)
        return logits, emb

# ------------------ train/val ------------------

def topk_acc(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)):
    maxk = max(ks)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = []
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out.append((correct_k / targets.size(0)).item())
    return out

def make_sampler(labels: List[int]):
    from collections import Counter
    counts = Counter(labels)
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def read_list(path: Path):
    paths, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            rel, idx = line.strip().split()
            paths.append(rel); labels.append(int(idx))
    return paths, labels


def main():
    ap = argparse.ArgumentParser(description='Train ArcFace/CosFace head on frozen DINOv2')
    ap.add_argument('--fold_root', type=Path, required=True, help='e.g., out/cv/fold_01')
    ap.add_argument('--dino_model', type=str, default='dinov2_vitl14')
    ap.add_argument('--image_size', type=int, default=518)
    ap.add_argument('--embed_dim', type=int, default=512)
    ap.add_argument('--loss', type=str, default='arcface', choices=['arcface', 'cosface'])
    ap.add_argument('--margin', type=float, default=0.4)
    ap.add_argument('--scale', type=float, default=30.0)

    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--no_hflip', action='store_true')

    ap.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--val_every', type=int, default=1)
    ap.add_argument('--early_patience', type=int, default=8)
    ap.add_argument('--save', type=Path, required=True)

    args = ap.parse_args()

    fold = args.fold_root
    train_list = fold / 'train_list.txt'
    val_list = fold / 'val_list.txt'

    # Build datasets
    ds_train = ListTxtDataset(train_list, fold, args.image_size, train=True, no_hflip=args.no_hflip)
    ds_val = ListTxtDataset(val_list, fold, args.image_size, train=False, no_hflip=True)

    # Count classes
    _, train_labels = read_list(train_list)
    num_classes = max(train_labels) + 1

    # Sampler & loaders
    sampler = make_sampler(train_labels)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = FinIDModel(args.dino_model, args.embed_dim, num_classes, args.loss, args.scale, args.margin, args.device)

    # Optim & sched
    optim = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_top1 = 0.0
    patience = args.early_patience
    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, ce_loss = 0, 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(model.device, non_blocking=True), yb.to(model.device, non_blocking=True)
            logits, _ = model(xb, yb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += yb.size(0)
            ce_loss += loss.item() * yb.size(0)
        scheduler.step()

        if epoch % args.val_every == 0:
            model.eval()
            v_total, v_loss, v_top1, v_top5 = 0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb, yb = xb.to(model.device), yb.to(model.device)
                    logits, _ = model(xb, yb)
                    loss = F.cross_entropy(logits, yb)
                    t1, t5 = topk_acc(logits, yb, ks=(1, 5))
                    bs = yb.size(0)
                    v_total += bs
                    v_loss += loss.item() * bs
                    v_top1 += t1 * bs
                    v_top5 += t5 * bs
            v_loss /= max(1, v_total)
            v_top1 /= max(1, v_total)
            v_top5 /= max(1, v_total)
            print(f"Epoch {epoch:03d} | train_ce {ce_loss/total:.4f} | val_ce {v_loss:.4f} | val@1 {v_top1:.4f} | val@5 {v_top5:.4f}")

            if v_top1 > best_top1:
                best_top1 = v_top1
                patience = args.early_patience
                ckpt = {
                    'dino_model': args.dino_model,
                    'image_size': args.image_size,
                    'embed_dim': args.embed_dim,
                    'num_classes': num_classes,
                    'loss': args.loss,
                    'margin': args.margin,
                    'scale': args.scale,
                    'state_dict_head': model.head.state_dict(),
                    'state_dict_margin': model.margin.state_dict(),
                }
                torch.save(ckpt, args.save)
                print(f"[saved] {args.save}")
            else:
                patience -= 1
                if patience <= 0:
                    print("Early stopping.")
                    break

    print(f"Best val@1: {best_top1:.4f}")

if __name__ == '__main__':
    main()
