from __future__ import annotations

from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector, _mask_replace

from typing import Tuple, TypeVar

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time

SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]


# class CustomPredictorCorrector(PredictorCorrector):

    # @torch.no_grad()
    # def _denoise(
    #     self,
    #     batch: Diffusable,
    #     mask: dict[str, torch.Tensor],
    #     record: bool = False,
    # ) -> SampleAndMeanAndMaybeRecords:
    #     """Denoise from a prior sample to a t=eps_t sample."""
    #     recorded_samples = None
    #     if record:
    #         recorded_samples = []
    #     for k in self._predictors:
    #         mask.setdefault(k, None)
    #     for k in self._correctors:
    #         mask.setdefault(k, None)
    #     mean_batch = batch.clone()

    #     # Decreasing timesteps from T to eps_t
    #     timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
    #     dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

    #     for i in tqdm(range(self.N), miniters=50, mininterval=5):
    #         # Set the timestep
    #         t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)
            
    #         noisy_sample = self.diffusion_module.corruption.sample_marginal(batch, t)

    #         print('Before adding noise to host', batch)
            
    #         batch = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])
            
    #         print('After adding noise to host', batch)
            
    #         # Corrector updates.
    #         if self._correctors:
    #             for _ in range(self._n_steps_corrector):
    #                 score = self._score_fn(batch, t)
    #                 fns = {
    #                     k: corrector.step_given_score for k, corrector in self._correctors.items()
    #                 }
    #                 samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
    #                     fns=fns,
    #                     broadcast={"t": t},
    #                     x=batch,
    #                     score=score,
    #                     batch_idx=self._multi_corruption._get_batch_indices(batch),
    #                 )
    #                 if record:
    #                     recorded_samples.append(batch.clone().to("cpu"))
    #                 batch, mean_batch = _mask_replace(
    #                     samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
    #                 )

    #         # Predictor updates
    #         score = self._score_fn(batch, t)
    #         predictor_fns = {
    #             k: predictor.update_given_score for k, predictor in self._predictors.items()
    #         }
    #         samples_means = apply(
    #             fns=predictor_fns,
    #             x=batch,
    #             score=score,
    #             broadcast=dict(t=t, batch=batch, dt=dt),
    #             batch_idx=self._multi_corruption._get_batch_indices(batch),
    #         )
    #         if record:
    #             recorded_samples.append(batch.clone().to("cpu"))
    #         batch, mean_batch = _mask_replace(
    #             samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
    #         )

    #     return batch, mean_batch, recorded_samples
    

class CustomGuidedPredictorCorrector(GuidedPredictorCorrector):
    
    @torch.no_grad()
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
        
        print('This is the batch0', batch0['pos'], batch0['atomic_numbers'])

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)
            
            noisy_sample = self.diffusion_module.corruption.sample_marginal(batch0, t)

            # # print('Before adding noise to host', batch['pos'], batch['atomic_numbers'])
            
            batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])
            
            # print('After adding noise to host', batch['pos'], batch['atomic_numbers'])
            # print('This is the batch0', batch0['pos'], batch0['atomic_numbers'])
            
            # print('\n\n\n')
            
            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )

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
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

        return batch, mean_batch, recorded_samples

    
class GuidedPredictorCorrectorRevertedOrder(GuidedPredictorCorrector):
    @torch.no_grad()
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

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)


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
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t+dt)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t+dt, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )

        return batch, mean_batch, recorded_samples


    
    
class CustomGuidedPredictorCorrectorRePaint(GuidedPredictorCorrector):
    
    def __init__(self, **kwargs):
        self.n_resample_steps = kwargs.pop('n_resample_steps', 1)
        
        super().__init__(**kwargs)
        

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        ignore_mask = {}
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        mean_batch = batch.clone()
        
        # ignore_mask = mask

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        batch0 = batch.clone()
        noisy_sample = self.diffusion_module.corruption.sample_marginal(
            batch0, torch.full((batch.get_batch_size(),), timesteps[0], device=self._device)
            )
        batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            for i_res in range(self.n_resample_steps):
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
                    samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=ignore_mask
                )

                # Corrector updates.
                if self._correctors:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t+dt)
                        fns = {
                            k: corrector.step_given_score for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                            fns=fns,
                            broadcast={"t": t+dt, "dt": dt},
                            x=batch,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(batch),
                        )
                        if record:
                            recorded_samples.append(batch.clone().to("cpu"))
                        # TR: Setting mask to None with the same reason as mentioned above
                        batch, mean_batch = _mask_replace(
                            samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=ignore_mask
                        )
                if i < self.N - 1 and True:
                    noisy_sample = self.diffusion_module.corruption.sample_marginal(batch0, t+dt if i < self.N - 1 else t)
                    batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])
                
                if i_res < self.n_resample_steps - 1 and i < self.N - 1:
                    print('Resampling')
                    # t_prev = torch.full((batch.get_batch_size(),), timesteps[i+1], device=self._device)
                    # if i == self.N -1:
                    #     t_prev2 = t_prev
                    # else:
                    #     t_prev2 = torch.full((batch.get_batch_size(),), timesteps[i+2], device=self._device)
                    
                    # Get the sqrt(sigma_{t-1} ** 2 - sigma_{t-2} ** 2)
                    only_for_z, sigma_t_prev_t_prev2, _ = self._predictors['pos']._get_coeffs(
                        x=batch['pos'],
                        t=t+dt, 
                        batch=batch, 
                        dt=dt,
                        batch_idx=self._multi_corruption._get_batch_indices(batch)['pos']
                    )
                    z = torch.randn_like(only_for_z)
                    batch['pos'] = self._multi_corruption.corruptions['pos'].wrap(
                        batch['pos'] + sigma_t_prev_t_prev2 * z
                    )

        return batch, mean_batch, recorded_samples


    
class CustomGuidedPredictorCorrectorRePaintV2(GuidedPredictorCorrector):
    
    def __init__(self, **kwargs):
        self.n_resample_steps = kwargs.pop('n_resample_steps', 1)
        
        super().__init__(**kwargs)
        

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        ignore_mask = {}
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        mean_batch = batch.clone()
        
        # ignore_mask = mask

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        batch0 = batch.clone()
        noisy_sample = self.diffusion_module.corruption.sample_marginal(
            batch0, torch.full((batch.get_batch_size(),), timesteps[0], device=self._device)
            )
        batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            for i_res in range(self.n_resample_steps):
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
                    samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=ignore_mask
                )

                # Corrector updates.
                if self._correctors:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t+dt)
                        fns = {
                            k: corrector.step_given_score for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                            fns=fns,
                            broadcast={"t": t+dt, "dt": dt},
                            x=batch,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(batch),
                        )
                        if record:
                            recorded_samples.append(batch.clone().to("cpu"))
                        # TR: Setting mask to None with the same reason as mentioned above
                        batch, mean_batch = _mask_replace(
                            samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=ignore_mask
                        )
                if i < self.N - 1 and True:
                    noisy_sample = self.diffusion_module.corruption.sample_marginal(batch0, t+dt if i < self.N - 1 else t)
                    batch['pos'] = batch['pos'].lerp_(noisy_sample['pos'], mask['pos'])
                
                if i_res < self.n_resample_steps - 1 and i < self.N - 1:
                    print('Resampling')
                    # t_prev = torch.full((batch.get_batch_size(),), timesteps[i+1], device=self._device)
                    # if i == self.N -1:
                    #     t_prev2 = t_prev
                    # else:
                    #     t_prev2 = torch.full((batch.get_batch_size(),), timesteps[i+2], device=self._device)
                    
                    # Get the sqrt(sigma_{t-1} ** 2 - sigma_{t-2} ** 2)
                    only_for_z, sigma_t_prev_t_prev2, _ = self._predictors['pos']._get_coeffs(
                        x=batch['pos'],
                        t=t+dt, 
                        batch=batch, 
                        dt=dt,
                        batch_idx=self._multi_corruption._get_batch_indices(batch)['pos']
                    )
                    z = torch.randn_like(only_for_z)
                    batch['pos'] = self._multi_corruption.corruptions['pos'].wrap(
                        batch['pos'] + sigma_t_prev_t_prev2 * z
                    )

        return batch, mean_batch, recorded_samples

    # def __init__(
    #     self,
    #     *,
    #     guidance_scale: float,
    #     remove_conditioning_fn: BatchTransform,
    #     keep_conditioning_fn: BatchTransform | None = None,
    #     **kwargs,
    # ):
    #     """
    #     guidance_scale: gamma in p_gamma(x|y)=p(x)p(y|x)**gamma for classifier-free guidance
    #     remove_conditioning_fn: function that removes conditioning from the data
    #     keep_conditioning_fn: function that will be applied to the data before evaluating the conditional score. For example, this function might drop some fields that you never want to condition on or add fields that indicate which conditions should be respected.
    #     **kwargs: passed on to parent class constructor.
    #     """

    #     super().__init__(**kwargs)
    #     self._remove_conditioning_fn = remove_conditioning_fn
    #     self._keep_conditioning_fn = keep_conditioning_fn or identity
    #     self._guidance_scale = guidance_scale