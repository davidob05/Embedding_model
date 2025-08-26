import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def list_classes(src: Path):
    # Top-level non-hidden dirs = classes (whale IDs)
    return sorted([d for d in src.iterdir() if d.is_dir() and not d.name.startswith(".")], key=lambda p: p.name.lower())

def collect_images_per_class(cls_dir: Path):
    # Recurse to handle nested structures
    return sorted([p for p in cls_dir.rglob("*") if is_image(p)], key=lambda p: p.name.lower())

def copy_many(files, dst_dir: Path, move=False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = dst_dir / f.name
        if move:
            shutil.move(str(f), str(dst))
        else:
            shutil.copy2(str(f), str(dst))

def write_list_txt(pairs, outpath: Path):
    """
    pairs: list of (relative_path, class_index)
    Writes lines like: "relative/path/to/img.jpg class_index"
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w") as f:
        for rel, idx in pairs:
            f.write(f"{rel} {idx}\n")

def build_kfold_indices(imgs, k, seed):
    """
    imgs: list[Path] for a single class
    Returns: list of length k; each element is (train_indices, val_indices)
    Strategy: shuffle once; then for fold i, val = [i % n] (round-robin), train = rest.
    If n == 0: k empty folds.
    If n == 1: in each fold, val has the lone image, train is empty (OK for evaluation; training code should handle).
    """
    rnd = random.Random(seed)
    order = imgs[:]  # copy
    rnd.shuffle(order)
    n = len(order)
    folds = []
    for i in range(k):
        if n == 0:
            folds.append(([], []))
            continue
        val_idx = [i % n]
        train_idx = [j for j in range(n) if j not in val_idx]
        folds.append((train_idx, val_idx))
    return order, folds

def main():
    ap = argparse.ArgumentParser(description="Organize whale dorsal fin photos for embedding training (K-fold CV + gallery/probe).")
    ap.add_argument("--src", type=Path, required=True, help="Parent folder containing subfolders per whale (class).")
    ap.add_argument("--out", type=Path, default=Path("out"), help="Output root for organized dataset.")
    ap.add_argument("--kfold", type=int, default=3, help="Number of CV folds (use 3 for tiny datasets; can set higher).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"[error] --src does not exist: {args.src}")

    classes = list_classes(args.src)
    if not classes:
        raise SystemExit(f"[error] No class subfolders found in: {args.src}")

    # Map class name to image list (shuffled per class)
    class_to_imgs = {}
    for cls in classes:
        imgs = collect_images_per_class(cls)
        class_to_imgs[cls.name] = imgs

    # Build class index mapping (stable, alphabetical)
    class_to_index = {cls.name: i for i, cls in enumerate(classes)}

    # ===== Create K-fold CV splits =====
    cv_root = args.out / "cv"
    k = max(1, args.kfold)

    # Prepare per-fold train/val file lists (for list.txt)
    fold_train_pairs = [list() for _ in range(k)]
    fold_val_pairs = [list() for _ in range(k)]

    for cls in classes:
        imgs = class_to_imgs[cls.name]
        order, folds = build_kfold_indices(imgs, k, args.seed)

        for i in range(k):
            train_idx, val_idx = folds[i]
            # Paths for this fold
            train_dir = cv_root / f"fold_{i+1:02d}" / "train" / cls.name
            val_dir   = cv_root / f"fold_{i+1:02d}" / "val"   / cls.name

            # Copy/Move
            train_files = [order[j] for j in train_idx]
            val_files   = [order[j] for j in val_idx]

            if train_files:
                copy_many(train_files, train_dir, move=args.move)
            if val_files:
                copy_many(val_files, val_dir, move=args.move)

            # Record relative paths for list files (relative to the fold root)
            # Build relative paths after files are in place
            for f in train_files:
                rel = (Path("train") / cls.name / f.name).as_posix()
                fold_train_pairs[i].append((rel, class_to_index[cls.name]))
            for f in val_files:
                rel = (Path("val") / cls.name / f.name).as_posix()
                fold_val_pairs[i].append((rel, class_to_index[cls.name]))

    # Write list files per fold
    for i in range(k):
        fold_root = cv_root / f"fold_{i+1:02d}"
        write_list_txt(fold_train_pairs[i], fold_root / "train_list.txt")
        write_list_txt(fold_val_pairs[i],   fold_root / "val_list.txt")

    # ===== Create a simple gallery/probe eval split =====
    # Choose 1 gallery image per class when available; the rest go to probe.
    eval_root = args.out / "eval"
    for cls in classes:
        imgs = class_to_imgs[cls.name]
        if not imgs:
            continue
        imgs_shuffled = imgs[:]
        random.Random(args.seed).shuffle(imgs_shuffled)
        gallery = [imgs_shuffled[0]]
        probe   = imgs_shuffled[1:] if len(imgs_shuffled) > 1 else []

        copy_many(gallery, eval_root / "gallery" / cls.name, move=args.move)
        if probe:
            copy_many(probe,   eval_root / "probe"   / cls.name, move=args.move)

    # ===== Summary =====
    total_imgs = sum(len(v) for v in class_to_imgs.values())
    nonempty_classes = sum(1 for v in class_to_imgs.values() if len(v) > 0)

    print("=== Done ===")
    print(f"Classes (whales): {len(classes)}  (with images: {nonempty_classes})")
    print(f"Total images found: {total_imgs}")
    print(f"Output root: {args.out.resolve()}")
    print(f"CV folds: {k}")
    print("Per-fold lists: out/cv/fold_xx/train_list.txt and val_list.txt")
    print("Eval split: out/eval/gallery and out/eval/probe")

if __name__ == "__main__":
    main()
