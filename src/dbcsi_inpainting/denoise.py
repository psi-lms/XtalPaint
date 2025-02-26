from typing import Tuple, TypeVar
import torch
from tqdm import tqdm
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.pc_sampler import _mask_replace

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time
SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]

#############################
## New implementation
#############################

def _denoise(
    self,
    batch: Diffusable,
    mask: dict[str, torch.Tensor],
    record: bool = False,
) -> SampleAndMeanAndMaybeRecords:
    """Denoise from a prior sample to a t=eps_t sample."""
    recorded_samples = None
    if record:
        recorded_samples = []
    for k in self._predictors:
        mask.setdefault(k, None)
    for k in self._correctors:
        mask.setdefault(k, None)
    mean_batch = batch.clone()

    # Decreasing timesteps from T to eps_t
    timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
    dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

    batch0 = batch.clone()
    noisy_sample = self.diffusion_module.corruption.sample_marginal(batch0, t)
    batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])

    n_resample_steps = 3

    for i in tqdm(range(self.N), miniters=50, mininterval=5):
        # Set the timestep
        t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

        for i_res in range(n_resample_steps):
            # Predictor updates
            score = self._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score for k, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))

            # TR: I set mask to None, so that the predictor steps are alos passed to the corrector
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=None
            )

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    # TR: Setting mask to None with the same reason as mentioned above
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=None
                    )
            noisy_sample = self.diffusion_module.corruption.sample_marginal(batch0, t)
            batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])
            
            if i_res < n_resample_steps - 1 and i < self.N - 1:
                t_prev = torch.full((batch.get_batch_size(),), timesteps[i+1], device=self._device)
                # if i == self.N -1:
                #     t_prev2 = t_prev
                # else:
                #     t_prev2 = torch.full((batch.get_batch_size(),), timesteps[i+2], device=self._device)
                
                # Get the sqrt(sigma_{t-1} ** 2 - sigma_{t-2} ** 2)
                _, sigma_t_prev_t_prev2, _ = self._predictors['pos']._get_coeffs(
                    x=batch,
                    t=t_prev, batch=batch, dt=dt,
                    batch_idx=self._multi_corruption._get_batch_indices(batch)
                )
                z = torch.randn_like(sigma_t_prev_t_prev2)
                batch['pos'] = batch['pos'] + sigma_t_prev_t_prev2 * z

    return batch, mean_batch, recorded_samples