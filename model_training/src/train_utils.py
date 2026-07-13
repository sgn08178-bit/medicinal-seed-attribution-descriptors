from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import SeedClassificationDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build_transforms(cfg: dict[str, Any], train: bool):
    aug = cfg.get("augmentation", {})
    img_size = int(cfg["img_size"])
    mean = cfg.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = cfg.get("imagenet_std", [0.229, 0.224, 0.225])
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        if aug.get("random_horizontal_flip", {}).get("enabled", True):
            ops.append(transforms.RandomHorizontalFlip(p=float(aug["random_horizontal_flip"].get("p", 0.5))))
        if aug.get("random_rotation", {}).get("enabled", True):
            ops.append(transforms.RandomRotation(degrees=float(aug["random_rotation"].get("degrees", 10))))
        if aug.get("color_jitter", {}).get("enabled", True):
            cj = aug["color_jitter"]
            ops.append(
                transforms.ColorJitter(
                    brightness=float(cj.get("brightness", 0.15)),
                    contrast=float(cj.get("contrast", 0.15)),
                    saturation=float(cj.get("saturation", 0.10)),
                    hue=float(cj.get("hue", 0.02)),
                )
            )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def make_loader(csv_or_df, cfg: dict[str, Any], train: bool) -> DataLoader:
    ds = SeedClassificationDataset(csv_or_df, build_transforms(cfg, train=train))
    return DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=train,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        worker_init_fn=seed_worker,
        generator=get_generator(int(cfg["seed"])),
    )


def criterion_from_cfg(cfg: dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))


def optimizer_from_cfg(model: nn.Module, cfg: dict[str, Any]):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def scheduler_from_cfg(optimizer, cfg: dict[str, Any]):
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    epochs = int(cfg["epochs"])
    min_lr = float(cfg.get("min_learning_rate", 1e-6))
    warmup = None
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: (e + 1) / warmup_epochs if e < warmup_epochs else 1.0,
        )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=min_lr,
    )
    return warmup, cosine


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds, trues = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        trues.extend(labels.detach().cpu().numpy().tolist())
    return total_loss / max(1, len(loader.dataset)), accuracy_score(trues, preds), preds, trues


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    cfg: dict[str, Any],
    checkpoint_path: str | Path,
    history_path: str | Path,
    log_path: str | Path,
    device,
    overwrite: bool = False,
):
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)
    log_path = Path(log_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [p for p in [checkpoint_path, history_path, log_path] if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing training outputs found. Refusing to overwrite without --overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )
    for path in existing:
        path.unlink()

    criterion = criterion_from_cfg(cfg)
    optimizer = optimizer_from_cfg(model, cfg)
    warmup, cosine = scheduler_from_cfg(optimizer, cfg)
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    patience = int(cfg["patience"])

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, int(cfg["epochs"]) + 1):
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device)
            if warmup is not None and epoch <= int(cfg.get("warmup_epochs", 0)):
                warmup.step()
            else:
                cosine.step()
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            history.append(row)
            line = json.dumps(row, ensure_ascii=False)
            print(line)
            log.write(line + "\n")
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                wait += 1
            if wait >= patience:
                break

    pd.DataFrame(history).to_csv(history_path, index=False)
    return {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss), "history_path": str(history_path), "checkpoint_path": str(checkpoint_path)}
