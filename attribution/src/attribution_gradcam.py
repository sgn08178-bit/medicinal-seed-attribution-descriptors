from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TargetLayerInfo:
    name: str
    module: torch.nn.Module
    output_shape: list[int]
    reason: str


def _candidate_names(model_name: str) -> list[str]:
    if model_name == "resnet50":
        return ["layer4.2", "layer4"]
    if model_name == "efficientnet_b0":
        return ["conv_head", "blocks.6", "blocks.5"]
    if model_name == "convnext_small":
        return ["stages.3.blocks.2", "stages.3.blocks.1", "stages.3"]
    return []


def _is_spatial_output(out) -> bool:
    if isinstance(out, (tuple, list)):
        out = out[0]
    return torch.is_tensor(out) and out.ndim == 4 and out.shape[-1] > 1 and out.shape[-2] > 1


def select_target_layer(model: torch.nn.Module, model_name: str, sample: torch.Tensor) -> TargetLayerInfo:
    modules = dict(model.named_modules())
    preferred = _candidate_names(model_name)
    names = [n for n in preferred if n in modules]
    names += [n for n, m in modules.items() if n and n not in names and len(list(m.children())) == 0]
    last_good = None
    for name in names:
        outputs = {}
        handle = modules[name].register_forward_hook(lambda _m, _i, o, key=name: outputs.setdefault(key, o))
        try:
            with torch.no_grad():
                _ = model(sample)
            out = outputs.get(name)
        finally:
            handle.remove()
        if _is_spatial_output(out):
            shape = list(out[0].shape if isinstance(out, (tuple, list)) else out.shape)
            last_good = TargetLayerInfo(
                name=name,
                module=modules[name],
                output_shape=shape,
                reason=f"Selected as classifier-preceding spatial feature candidate for {model_name}.",
            )
            if name in preferred:
                return last_good
    if last_good is None:
        raise RuntimeError(f"No spatial target layer found for {model_name}.")
    return last_good


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, relu: bool = True):
        self.model = model
        self.target_layer = target_layer
        self.relu = relu
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output[0] if isinstance(output, (tuple, list)) else output

    def _backward_hook(self, _module, _grad_input, grad_output):
        grad = grad_output[0]
        self.gradients = grad[0] if isinstance(grad, (tuple, list)) else grad

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, images: torch.Tensor, targets: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        score = logits.gather(1, targets.view(-1, 1)).sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        if self.relu:
            cam = F.relu(cam)
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        cams = cam.squeeze(1).detach().cpu().numpy().astype(np.float32)
        return cams

