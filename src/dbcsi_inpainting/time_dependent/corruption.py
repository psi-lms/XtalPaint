"""TD-Paint corruption module."""

import torch
from mattergen.diffusion.corruption.corruption import (
    B,
    BatchedData,
    _broadcast_like,
    maybe_expand,
)
from mattergen.diffusion.corruption.sde_lib import SDE as DiffSDE
from mattergen.diffusion.corruption.sde_lib import VESDE as DiffVESDE
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE


class TDNumAtomsVarianceAdjustedWrappedVESDE(WrappedVESDE):
    """Wrapped VESDE with variance adjusted by number of atoms.

    We divide the standard deviation by the cubic root of the number of atoms.
    The goal is to reduce the influence by the cell size on the variance of the
    fractional coordinates.
    """

    def __init__(
        self,
        wrapping_boundary: float | torch.Tensor = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        limit_info_key: str = "num_atoms",
    ):
        """Initialize the NumAtomsVarianceAdjustedWrappedVESDE."""
        super().__init__(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            wrapping_boundary=wrapping_boundary,
        )
        self.limit_info_key = limit_info_key

    def std_scaling(self, batch: BatchedData) -> torch.Tensor:
        """Get the standard deviation scaling factor."""
        std_scale = batch[self.limit_info_key] ** (-1 / 3)
        return std_scale[batch.get_batch_idx("pos")]

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and std of the perturbation kernel."""
        assert t.shape[0] == x.shape[0]

        mean, std = super().marginal_prob(x, t, None, batch)
        assert batch is not None, (
            "batch must be provided when using "
            "NumAtomsVarianceAdjustedWrappedVESDEMixin"
        )
        std_scale = self.std_scaling(batch)
        std_scale = _broadcast_like(std_scale, std)

        assert t.shape[0] == std_scale.shape[0]
        assert std_scale.shape[0] == std.shape[0]

        std = std * std_scale  # maybe_expand(std_scale, batch_idx, like=std)

        return mean, std

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx=None,
    ) -> torch.Tensor:
        """Generate prior samples with variance adjusted by number of atoms."""
        _super = super()
        assert isinstance(self, DiffSDE) and hasattr(_super, "prior_sampling")
        assert conditioning_data is not None, (
            "batch must be provided when using "
            "NumAtomsVarianceAdjustedWrappedVESDEMixin"
        )
        num_atoms = conditioning_data[self.limit_info_key]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_atoms.shape[0], device=num_atoms.device),
            num_atoms,
            dim=0,
        )
        std_scale = self.std_scaling(conditioning_data)
        # prior sample is randn() * sigma_max, so we need additionally multiply
        # by std_scale to get the correct variance.
        # We call VESDE.prior_sampling (a "grandparent" function) because the
        # super() prior_sampling already does the wrapping, which means we
        # couldn't do the variance adjustment here anymore otherwise.
        prior_sample = DiffVESDE.prior_sampling(self, shape=shape).to(
            num_atoms.device
        )
        return self.wrap(
            prior_sample
            * maybe_expand(std_scale, batch_idx, like=prior_sample)
        )

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the drift and diffusion of the SDE."""
        sigma = self.marginal_prob(x, t, batch_idx, batch)[1]
        sigma_min = self.marginal_prob(
            x, torch.zeros_like(t), batch_idx, batch
        )[1]
        sigma_max = self.marginal_prob(
            x, torch.ones_like(t), batch_idx, batch
        )[1]
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(2 * (sigma_max.log() - sigma_min.log()))
        return drift, diffusion
