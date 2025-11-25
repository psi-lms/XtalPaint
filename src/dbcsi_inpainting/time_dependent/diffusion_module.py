"""TD-Paint Diffusion Module."""

from typing import Callable, TypeVar

import torch
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule

T = TypeVar("T", bound=BatchedData)
BatchTransform = Callable[[T], T]


def identity(x: T) -> T:
    """Identity transform for batches."""
    return x


class TDDiffusionModule(DiffusionModule):
    """TD-Paint Diffusion Module."""

    def __init__(
        self,
        t_replace: float = 0.005,
        p_replace: float | None = None,
        **kwargs,
    ):
        """Initialize the TD diffusion module."""
        super().__init__(**kwargs)

        self.t_replace = t_replace
        self.p_replace = p_replace

    def sample_timesteps(self, batch: T) -> torch.Tensor:
        """Sample the timesteps.

        Args:
           batch: batch of data to be corrupted

        Returns: sampled timesteps
        """
        timesteps = self.timestep_sampler(
            batch.get_batch_size(),
            device=self._get_device(batch),
        )
        if self.p_replace is not None:
            timesteps = timesteps[batch.get_batch_idx("pos")]

            timesteps = timesteps.flatten()
            index_set_to_min = (
                torch.rand(batch["pos"].shape[0]) < self.p_replace
            )

            timesteps[index_set_to_min] = self.t_replace

        breakpoint()
        return timesteps
