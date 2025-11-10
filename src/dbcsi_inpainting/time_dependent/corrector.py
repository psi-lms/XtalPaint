import torch
from torch_scatter import scatter_add

from mattergen.diffusion.corruption.corruption import maybe_expand
import mattergen.diffusion.sampling.predictors_correctors as pc
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.wrapped.wrapped_sde import WrappedSDEMixin
from mattergen.diffusion.sampling.predictors_correctors import (
    LangevinCorrector,
)
from mattergen.diffusion.corruption import sde_lib

SampleAndMean = tuple[torch.Tensor, torch.Tensor]


class TDLangevinCorrector(LangevinCorrector):
    def step_given_score(
        self,
        *,
        x,
        batch_idx: torch.LongTensor | None,
        score,
        t: torch.Tensor,
        dt: torch.Tensor,
        mask: torch.LongTensor | None = None,
    ) -> SampleAndMean:
        if mask is None:
            mask = torch.zeros(x.shape[0])
        mask_bool = ~mask.bool()

        alpha = self.get_alpha(t, dt=dt)
        snr = self.snr
        noise = torch.randn_like(score)
        breakpoint()
        grad_norm_square = (
            torch.square(score).reshape(score.shape[0], -1).sum(dim=1)
        )
        noise_norm_square = (
            torch.square(noise).reshape(noise.shape[0], -1).sum(dim=1)
        )
        breakpoint()
        if batch_idx is None:
            grad_norm = grad_norm_square.sqrt().mean()
            breakpoint()
            noise_norm = noise_norm_square.sqrt().mean()
        else:
            grad_norm = torch.sqrt(
                scatter_add(
                    grad_norm_square[mask_bool],
                    dim=-1,
                    index=batch_idx[mask_bool],
                )
            ).mean()
            breakpoint()
            noise_norm = torch.sqrt(
                scatter_add(
                    noise_norm_square[mask_bool],
                    dim=-1,
                    index=batch_idx[mask_bool],
                )
            ).mean()
        breakpoint()

        # If gradient is zero (i.e., we are sampling from an improper
        # distribution that's flat over the whole of R^n)
        # the step_size blows up. Clip step_size to avoid this.
        # The EGNN reports zero scores when there are no edges between nodes.
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, self.max_step_size)
        step_size[grad_norm == 0, :] = self.max_step_size

        # Expand step size to batch structure (score and noise have
        # the same shape).
        step_size = maybe_expand(step_size, batch_idx, score)
        breakpoint()
        # Perform update, using custom update for SO(3) diffusion on frames.
        mean = x + step_size * score
        x = mean + torch.sqrt(step_size * 2) * noise

        return x, mean


class TDWrappedCorrectorMixin:
    """A mixin for wrapping the corrector in a WrappedSDE."""

    def step_given_score(
        self,
        *,
        x: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        mask: torch.LongTensor | None,
    ) -> SampleAndMean:
        # mypy
        assert isinstance(self, pc.LangevinCorrector)
        _super = super()
        assert hasattr(_super, "step_given_score")
        assert hasattr(self, "corruption") and hasattr(self.corruption, "wrap")
        if not hasattr(self.corruption, "wrap"):
            raise IncompatibleSampler(
                f"{self.__class__.__name__} is not compatible "
                f"with {self.corruption}."
            )
        sample, mean = _super.step_given_score(
            x=x, score=score, t=t, batch_idx=batch_idx, dt=dt, mask=mask
        )
        return self.corruption.wrap(sample), self.corruption.wrap(mean)


class TDWrappedLangevinCorrector(TDWrappedCorrectorMixin, TDLangevinCorrector):
    @classmethod
    def is_compatible(cls, corruption: Corruption):
        return isinstance(
            corruption, (sde_lib.VPSDE, sde_lib.VESDE)
        ) and isinstance(corruption, WrappedSDEMixin)
