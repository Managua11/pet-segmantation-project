import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import MEAN, STD, PetDataset, get_splits
from src.metrics import dice_score, iou_score
from src.predict import load_model

DATA_ROOT = "data/oxford-iiit-pet"


def read_species():
    species = {}
    for fname in ["trainval.txt", "test.txt"]:
        with open(os.path.join(DATA_ROOT, "annotations", fname)) as f:
            for line in f:
                parts = line.split()
                if parts and not parts[0].startswith("#"):
                    species[parts[0]] = "cat" if parts[2] == "1" else "dog"
    return species


def test_dataset(img_size):
    masks_dir = os.path.join(DATA_ROOT, "annotations", "trimaps")
    with open(os.path.join(DATA_ROOT, "annotations", "test.txt")) as f:
        names = [line.split()[0] for line in f if line.strip()]
    valid = []
    for n in names:
        mask = cv2.imread(os.path.join(masks_dir, f"{n}.png"), cv2.IMREAD_UNCHANGED)
        if mask is not None and (mask != 2).any():
            valid.append(n)
    print(f"ამოღებულია ცარიელი/არავალიდური ნიღაბი: {len(names) - len(valid)}")
    images = [os.path.join(DATA_ROOT, "images", f"{n}.jpg") for n in valid]
    masks = [os.path.join(masks_dir, f"{n}.png") for n in valid]
    return PetDataset(images, masks, img_size=img_size, augment=False)


def size_bucket(fraction):
    if fraction < 0.2:
        return "small"
    return "medium" if fraction <= 0.5 else "large"


def save_worst(model, ds, indices, ious, path):
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    for row, i in enumerate(indices):
        image, mask = ds[i]
        with torch.no_grad():
            pred = (torch.sigmoid(model(image.unsqueeze(0)))[0, 0] > 0.5).numpy()
        rgb = np.clip(image.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        name = os.path.basename(ds.image_paths[i])
        for ax, img, title in zip(axes[row], [rgb, mask[0].numpy(), pred],
                                  [f"{name} IoU={ious[i]:.3f}", "ground truth", "prediction"]):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    model, img_size = load_model(args.checkpoint)
    ds = get_splits(DATA_ROOT, img_size=img_size)[1] if args.split == "val" else test_dataset(img_size)
    species = read_species()

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            image, mask = ds[i]
            logits = model(image.unsqueeze(0))
            mask = mask.unsqueeze(0)
            name = os.path.splitext(os.path.basename(ds.image_paths[i]))[0]
            rows.append({"name": name, "species": species[name], "breed": name.rsplit("_", 1)[0],
                         "pet_fraction": mask.mean().item(),
                         "iou": iou_score(logits, mask), "dice": dice_score(logits, mask)})
    df = pd.DataFrame(rows)

    exp_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    out_dir = os.path.join("experiments", exp_name, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)

    df["size"] = df["pet_fraction"].apply(size_bucket)
    summary = {
        "n_images": len(df),
        "mean_iou": df["iou"].mean(),
        "mean_dice": df["dice"].mean(),
        "median_iou": df["iou"].median(),
        "iou_by_species": df.groupby("species")["iou"].mean().to_dict(),
        "iou_by_size": df.groupby("size")["iou"].mean().to_dict(),
        "worst_breeds": df.groupby("breed")["iou"].mean().nsmallest(5).to_dict(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    worst = df["iou"].nsmallest(10).index.tolist()
    save_worst(model, ds, worst, df["iou"].tolist(), os.path.join(out_dir, "worst_10.png"))

    print(json.dumps(summary, indent=2))
    print(f"შედეგები: {out_dir}")


if __name__ == "__main__":
    main()
