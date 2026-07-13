from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import timm
import torch
from torch import nn


MODEL_ALIASES = {
    "convnext_small": "convnext_small",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
}


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if model_name not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model_name={model_name}")
    return timm.create_model(MODEL_ALIASES[model_name], pretrained=pretrained, num_classes=num_classes)


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in state:
                state = state[key]
                break
    if isinstance(state, OrderedDict):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_logits(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model(images)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return logits, pred, conf

