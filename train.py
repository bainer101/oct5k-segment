# train.py
# Usage examples:
#   python train.py --data_root /path/to/OCT5k --num_classes 7 --model unet --epochs 60
#   python train.py --data_root /path/to/OCT5k --model deeplabv3 --backbones resnet50 mobilenetv2 xception --losses cce dice focal dice+ce --epochs 80
#   python train.py --data_root /path/to/OCT5k --model deeplabv3plus --resize 348 348 --batch_size 8

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

# Local imports
from datasets.OCT5k import OCT5kDataset
from models.unet.unet_model import UNet
from models.deeplabv3 import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet,
    deeplabv3_xception,
    deeplabv3_octf,
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet101,
    deeplabv3plus_mobilenet,
    deeplabv3plus_xception,
    deeplabv3plus_octf
)

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    return batch.to(device, non_blocking=True)


class ChannelAdapter(nn.Module):
    """
    Ensures input has 3 channels for backbones that expect RGB.
    If in_ch == 1 -> repeat to 3; if in_ch == 3 -> identity; else -> 1x1 conv to 3.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch
        if in_ch == 3:
            self.adapt = nn.Identity()
        elif in_ch == 1:
            self.adapt = None  # handled in forward by repeat
        else:
            self.adapt = nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)

    def forward(self, x):
        if self.in_ch == 3:
            return x
        if self.in_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return self.adapt(x)
    

# add near ChannelAdapter
class ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, x):  # x is (B,3,H,W)
        return (x - self.mean) / self.std


# ---------------------------
# Losses
# ---------------------------

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int = None):
        super().__init__()
        self.C = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (B,C,H,W), target: (B,H,W) long
        probs = F.softmax(logits, dim=1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            target = target.clone()
            target[~valid] = 0
        target_1h = F.one_hot(target, num_classes=self.C).permute(0,3,1,2).float()
        if self.ignore_index is not None:
            mask = valid.float().unsqueeze(1)
            probs = probs * mask
            target_1h = target_1h * mask

        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        # mean over classes
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        if self.ignore_index is None:
            ce = F.cross_entropy(logits, target, reduction="none")
        else:
            ce = F.cross_entropy(logits, target, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.ignore_index is not None:
            focal = focal[target != self.ignore_index]
        return focal.mean()


class TverskyLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0, ignore_index: int = None):
        super().__init__()
        self.C = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            target = target.clone()
            target[~valid] = 0
        target_1h = F.one_hot(target, num_classes=self.C).permute(0,3,1,2).float()
        if self.ignore_index is not None:
            mask = valid.float().unsqueeze(1)
            probs = probs * mask
            target_1h = target_1h * mask

        dims = (0, 2, 3)
        tp = (probs * target_1h).sum(dims)
        fp = (probs * (1 - target_1h)).sum(dims)
        fn = ((1 - probs) * target_1h).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


def _ce_loss(ignore_index):
    return nn.CrossEntropyLoss() if ignore_index is None else nn.CrossEntropyLoss(ignore_index=ignore_index)

def make_loss(name: str, num_classes: int, ignore_index: int = None):
    name = name.lower()
    if name in ["cce", "ce", "crossentropy", "cross-entropy"]:
        return _ce_loss(ignore_index)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(ignore_index=ignore_index)
    if name in ["dice+ce", "dice_ce", "combo"]:
        return ComboLoss(
            _ce_loss(ignore_index),
            DiceLoss(num_classes=num_classes, ignore_index=ignore_index),
            w_ce=0.5, w_dice=0.5
        )
    if name == "tversky":
        return TverskyLoss(num_classes=num_classes, alpha=0.5, beta=0.5, ignore_index=ignore_index)
    raise ValueError(f"Unknown loss: {name}")


class ComboLoss(nn.Module):
    def __init__(self, ce_loss: nn.Module, dice_loss: nn.Module, w_ce: float = 0.5, w_dice: float = 0.5):
        super().__init__()
        self.ce = ce_loss
        self.dice = dice_loss
        self.w_ce = w_ce
        self.w_dice = w_dice

    def forward(self, logits, target):
        return self.w_ce * self.ce(logits, target) + self.w_dice * self.dice(logits, target)


# ---------------------------
# Metrics
# ---------------------------
def _update_confusion(conf: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, C: int, ignore_index: int = None):
    # preds, target are (N,) 1D longs on same device as conf
    if ignore_index is not None:
        keep = target != ignore_index
        preds = preds[keep]
        target = target[keep]
    if target.numel() == 0:
        return
    inds = C * target + preds
    conf += torch.bincount(inds, minlength=C * C).reshape(C, C)


@torch.no_grad()
def metrics_from_confusion(conf: torch.Tensor) -> dict:
    # conf: CxC with rows=gt, cols=pred
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    denom_iou = tp + fp + fn
    iou_c = torch.where(denom_iou > 0, tp / denom_iou.clamp(min=1), torch.zeros_like(tp))
    dice_c = torch.where((2 * tp + fp + fn) > 0, (2 * tp) / (2 * tp + fp + fn).clamp(min=1), torch.zeros_like(tp))
    acc = (tp.sum() / conf.sum().clamp(min=1)).item()
    return {
        "dice_per_class": dice_c.cpu().tolist(),
        "iou_per_class": iou_c.cpu().tolist(),
        "dice_mean": dice_c.mean().item(),
        "iou_mean": iou_c.mean().item(),
        "acc": acc,
        "support_per_class": conf.sum(1).long().cpu().tolist(),  # number of gt pixels per class
    }



# ---------------------------
# Models
# ---------------------------

def build_model(model_name: str, backbone: str, in_channels: int, num_classes: int):
    model_name = model_name.lower()
    if model_name == "unet":
        model = UNet(n_channels=in_channels, n_classes=num_classes, bilinear=True)
        adapter = nn.Identity()  # UNet accepts 1-ch natively
        return nn.Sequential(adapter, model)  # unify interface
    # DeepLab variants:
    # Always ensure 3-ch input via ChannelAdapter
    adapter = ChannelAdapter(in_ch=in_channels)
    norm=ImageNetNorm()  # normalize to ImageNet stats after adapting channels
    if model_name == "deeplabv3":
        if backbone == "resnet50":
            core = deeplabv3_resnet50(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "resnet101":
            core = deeplabv3_resnet101(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "mobilenetv2":
            core = deeplabv3_mobilenet(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "xception":
            core = deeplabv3_xception(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "octf":
            core = deeplabv3_octf(num_classes=num_classes, output_stride=16, pretrained_backbone=True)
            
        else:
            raise ValueError(f"Unsupported backbone for deeplabv3: {backbone}")
    elif model_name == "deeplabv3plus":
        if backbone == "resnet50":
            core = deeplabv3plus_resnet50(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "resnet101":
            core = deeplabv3plus_resnet101(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "mobilenetv2":
            core = deeplabv3plus_mobilenet(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "xception":
            core = deeplabv3plus_xception(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
        elif backbone == "octf":
            core = deeplabv3plus_octf(num_classes=num_classes, output_stride=16, pretrained_backbone=True)
        else:
            raise ValueError(f"Unsupported backbone for deeplabv3plus: {backbone}")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return nn.Sequential(adapter, norm, core)


# ---------------------------
# Trainer
# ---------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        set_seed(args.seed)

        # Transforms
        img_tf, msk_tf = self._make_transforms(args.resize)

        # Datasets / Loaders
        self.train_ds = OCT5kDataset(args.data_root, split="train", img_transform=img_tf, msk_transform=msk_tf)
        self.val_ds   = OCT5kDataset(args.data_root, split="valid", img_transform=img_tf, msk_transform=msk_tf)
        self.test_ds  = OCT5kDataset(args.data_root, split="test",  img_transform=img_tf, msk_transform=msk_tf)

        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.workers, pin_memory=True)
        self.test_loader  = DataLoader(self.test_ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.workers, pin_memory=True)

        self.scaler = GradScaler(enabled=(self.device.type == "cuda" and not args.no_amp))

        # Logging / checkpoints
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "runs.csv"
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "backbone", "loss", "best_epoch",
                    "val_dice", "val_iou", "val_acc",
                    "val_dice_per_class", "val_iou_per_class", "val_support_per_class",
                    "test_dice", "test_iou", "test_acc",
                    "test_dice_per_class", "test_iou_per_class", "test_support_per_class",
                    "ckpt_path"
                ])


    def _make_transforms(self, resize):
        if resize is None:
            img_tf = transforms.Compose([
                transforms.ToTensor(),  # (1,H,W) float [0,1], PIL 'L' -> 1 channel tensor
            ])
            msk_tf = transforms.Compose([
                transforms.PILToTensor(),  # (1,H,W) uint8
                transforms.Lambda(lambda t: t.squeeze(0).long())
            ])
        else:
            H, W = resize
            img_tf = transforms.Compose([
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
            msk_tf = transforms.Compose([
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.PILToTensor(),
                transforms.Lambda(lambda t: t.squeeze(0).long())
            ])
        return img_tf, msk_tf

    def run(self):
        # NOTE: backbones only apply to DeepLab; UNet ignores them
        for model_name in self.args.models:
            if model_name == "unet":
                model_backbones = ["n/a"]
            else:
                model_backbones = self.args.backbones

            for backbone in model_backbones:
                for loss_name in self.args.losses:
                    print(f"\n=== Run: model={model_name} backbone={backbone} loss={loss_name} ===")
                    model = build_model(model_name, backbone, in_channels=1, num_classes=self.args.num_classes).to(self.device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=0.5,
                        patience=max(1, self.args.patience // 2), verbose=True
                    )
                    criterion = make_loss(loss_name, num_classes=self.args.num_classes, ignore_index=self.args.ignore_index)

                    best = {"epoch": -1, "dice": -1.0, "iou": -1.0, "acc": -1.0}
                    epochs_no_improve = 0
                    ckpt_dir = self.out_dir / f"{model_name}_{backbone}_{loss_name}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_best = ckpt_dir / "best.pt"

                    for epoch in range(1, self.args.epochs + 1):
                        train_loss = self._train_one_epoch(model, criterion, optimizer, epoch)
                        val_metrics = self._eval(model, self.val_loader)
                        scheduler.step(val_metrics["dice_mean"])

                        print(f"[{model_name}/{backbone}/{loss_name}] "
                            f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  "
                            f"val_dice={val_metrics['dice_mean']:.4f}  "
                            f"val_iou={val_metrics['iou_mean']:.4f}  val_acc={val_metrics['acc']:.4f}")

                        if val_metrics["dice_mean"] > best["dice"]:
                            best.update({
                                "epoch": epoch,
                                "dice": val_metrics["dice_mean"],
                                "iou": val_metrics["iou_mean"],
                                "acc": val_metrics["acc"],
                            })
                            torch.save({
                                "epoch": epoch,
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "args": vars(self.args),
                                "val_metrics": val_metrics,
                                "model_name": model_name,
                                "backbone": backbone,
                                "loss_name": loss_name,
                            }, ckpt_best)
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1

                        if epochs_no_improve >= self.args.patience:
                            print(f"Early stopping (no improvement for {self.args.patience} epochs).")
                            break

                    if ckpt_best.exists():
                        state = torch.load(ckpt_best, map_location=self.device)
                        model.load_state_dict(state["model_state"])
                    test_metrics = self._eval(model, self.test_loader)

                    print(f"==> Best@epoch {best['epoch']}: val Dice={best['dice']:.4f}, IoU={best['iou']:.4f}")
                    print(f"==> Test: Dice={test_metrics['dice_mean']:.4f}, IoU={test_metrics['iou_mean']:.4f}, Acc={test_metrics['acc']:.4f}")

                    with open(self.log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            model_name, backbone, loss_name, best["epoch"],
                            round(best["dice"], 6), round(best["iou"], 6), round(best["acc"], 6),
                            json.dumps(state["val_metrics"]["dice_per_class"] if ckpt_best.exists() else []),
                            json.dumps(state["val_metrics"]["iou_per_class"] if ckpt_best.exists() else []),
                            json.dumps(state["val_metrics"]["support_per_class"] if ckpt_best.exists() else []),
                            round(test_metrics["dice_mean"], 6), round(test_metrics["iou_mean"], 6), round(test_metrics["acc"], 6),
                            json.dumps(test_metrics["dice_per_class"]),
                            json.dumps(test_metrics["iou_per_class"]),
                            json.dumps(test_metrics["support_per_class"]),
                            str(ckpt_best.resolve())
                        ])

    def _train_one_epoch(self, model, criterion, optimizer, epoch: int) -> float:
        model.train()
        running = 0.0
        n = 0
        for imgs, msks in self.train_loader:
            imgs, msks = to_device(imgs, self.device), to_device(msks, self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=(self.device.type == "cuda" and not self.args.no_amp)):
                logits = model(imgs)
                loss = criterion(logits, msks)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return running / max(1, n)

    @torch.no_grad()
    def _eval(self, model, loader) -> Dict[str, float]:
        model.eval()
        C = self.args.num_classes
        conf = torch.zeros((C, C), dtype=torch.long, device=self.device)

        for imgs, msks in loader:
            imgs, msks = to_device(imgs, self.device), to_device(msks, self.device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).view(-1)
            target = msks.view(-1)
            _update_confusion(conf, preds, target, C=C, ignore_index=self.args.ignore_index)

        return metrics_from_confusion(conf)



# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="General trainer for OCT5k on UNet / DeepLab (multi-backbone, multi-loss).")
    p.add_argument("--data_root", type=Path, required=True, help="OCT5k root with train_img/train_msk/valid_img/valid_msk/test_img/test_msk")
    p.add_argument("--out_dir", type=Path, default=Path("checkpoints"), help="Where to save runs and checkpoints")
    p.add_argument(
        "--models", type=str, nargs="+",
        choices=["unet", "deeplabv3", "deeplabv3plus"],
        help="Models to train. If omitted and --all_models not set, defaults to UNet."
    )
    p.add_argument(
        "--all_models", action="store_true",
        help="Shortcut for --models unet deeplabv3 deeplabv3plus"
    )

    p.add_argument("--backbones", type=str, nargs="*", default=["resnet50", "resnet101", "mobilenetv2", "xception", "octf"],
                   help="Backbones to try (ignored for UNet)")
    p.add_argument("--losses", type=str, nargs="*", default=["cce", "dice", "focal", "dice+ce"],
                   help="Which loss functions to compare")
    p.add_argument("--num_classes", type=int, required=True, help="Number of segmentation classes (mask IDs)")
    p.add_argument("--ignore_index", type=int, default=None, help="Class ID to ignore in loss/metrics")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without val Dice improvement)")
    p.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=[348, 348], help="Resize (H W). Use 348 348 for OCT5k.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_models:
        args.models = ["unet", "deeplabv3", "deeplabv3plus"]
    elif not args.models:
        args.models = ["unet"]
    Trainer(args).run()
