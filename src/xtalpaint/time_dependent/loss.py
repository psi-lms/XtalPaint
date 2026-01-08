"""TD-Paint loss functions for materials diffusion models."""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/wrapped/wrapped_normal_loss.py

from functools import partial
from typing import Dict, Literal, Optional

import torch
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import (
    SummedFieldLoss,
    denoising_score_matching,
)
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import (
    FieldLoss,
    aggregate_per_sample,
    d3pm_loss,
)
from mattergen.diffusion.wrapped.wrapped_normal_loss import (
    wrapped_normal_score,
)


def wrapped_normal_loss_td(
    *,
    corruption: SDE,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: Optional[torch.LongTensor],
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    batch: BatchedData,
    **_,
) -> torch.Tensor:
    """Compute the loss for a wrapped normal distribution.

    Compares the score of the wrapped normal distribution to the score of
    the score model.
    """
    assert len(t) == x.shape[0]
    _, std = corruption.marginal_prob(
        x=torch.zeros((x.shape[0], 1), device=t.device),
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )  # std does not depend on x

    pred: torch.Tensor = score_model_output
    if pred.ndim != 2:
        raise NotImplementedError

    assert hasattr(corruption, "wrapping_boundary"), (
        "SDE must be a WrappedSDE, i.e., must have a wrapping boundary."
    )
    wrapping_boundary = corruption.wrapping_boundary
    # Scaled identity matrix, i.e., in each dimension we wrap at
    # `wrapping_boundary`.
    wrapping_boundary = wrapping_boundary * torch.eye(
        x.shape[-1], device=t.device
    )[None].expand(batch_size, -1, -1)

    # We multiply the score by the standard deviation because we don't use
    # raw_noise here; raw_noise is -score * std, i.e., we multiply the
    # score by std.
    target = (
        wrapped_normal_score(
            x=noisy_x,
            mean=x,
            wrapping_boundary=wrapping_boundary,
            variance_diag=std.squeeze() ** 2,
            batch=batch_idx,
        )
        * std
    )
    delta = target - pred

    losses = delta.square()

    return aggregate_per_sample(
        losses, batch_idx, reduce=reduce, batch_size=batch_size
    )


class TDMaterialsLoss(SummedFieldLoss):
    """TD-Paint loss for materials diffusion models."""

    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the TDMaterialsLoss."""
        model_targets = {
            "pos": ModelTarget.score_times_std,
            "cell": ModelTarget.score_times_std,
        }
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss_td,
                reduce=reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")
            loss_fns["atomic_numbers"] = partial(
                d3pm_loss,
                reduce=reduce,
                d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            )
        self.reduce = reduce
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            model_targets=model_targets,
        )
