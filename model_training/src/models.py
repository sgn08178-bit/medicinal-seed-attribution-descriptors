from __future__ import annotations

import timm
import torch
from torch import nn


MODEL_ALIASES = {
    "convnext_small": "convnext_small",
    "ConvNeXt-Small": "convnext_small",
    "resnet50": "resnet50",
    "ResNet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "EfficientNet-B0": "efficientnet_b0",
}


def canonical_model_name(model_name: str) -> str:
    if model_name not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model_name={model_name}. Supported: {sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[model_name]


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(canonical_model_name(model_name), pretrained=pretrained, num_classes=num_classes)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model
