from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients


def compute_ig_for_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int,
    internal_batch_size: int | None,
    multiply_by_inputs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ig = IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs)
    baseline = torch.zeros_like(images)
    attrs, delta = ig.attribute(
        images,
        baselines=baseline,
        target=targets,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    attrs_2d = attrs.detach().abs().sum(dim=1).cpu().numpy()
    return attrs_2d.astype(np.float32), delta.detach().cpu().numpy()

