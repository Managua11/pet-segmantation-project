import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import get_splits
from src.metrics import dice_score, iou_score
from src.model import build_model

DATA_ROOT = "data/oxford-iiit-pet"
IMG_SIZE = 256
SEED = 42
ENCODER = "resnet18"


def set_train_mode(model, freeze_encoder):
    model.train()
    if freeze_encoder:
        model.encoder.eval()


def evaluate(model, loader, loss_fn):
    model.eval()
    loss_sum, iou_sum, dice_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images)
            b = images.size(0)
            loss_sum += loss_fn(logits, masks).item() * b
            iou_sum += iou_score(logits, masks) * b
            dice_sum += dice_score(logits, masks) * b
            n += b
    return loss_sum / n, iou_sum / n, dice_sum / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_ds, val_ds = get_splits(DATA_ROOT, seed=SEED, img_size=IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(ENCODER)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    exp_dir = os.path.join("experiments", args.exp_name)
    ckpt_dir = os.path.join("artifacts", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_iou,val_dice,epoch_time_sec\n")

    ckpt_args = {"exp_name": args.exp_name, "encoder": ENCODER, "epochs": args.epochs, "lr": args.lr,
                 "batch_size": args.batch_size, "img_size": IMG_SIZE,
                 "freeze_encoder": args.freeze_encoder, "seed": SEED}
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        set_train_mode(model, args.freeze_encoder)
        loss_sum, n = 0.0, 0
        for images, masks in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = loss_sum / n

        val_loss, val_iou, val_dice = evaluate(model, val_loader, loss_fn)
        epoch_time = time.time() - start

        print(f"epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_iou {val_iou:.4f} | val_dice {val_dice:.4f} | {epoch_time:.1f}s")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_iou:.6f},{val_dice:.6f},{epoch_time:.1f}\n")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": val_iou, "args": ckpt_args},
                       os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> best.pt saved (val_iou {val_iou:.4f})")


if __name__ == "__main__":
    main()
