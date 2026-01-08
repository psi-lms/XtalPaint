"""Predictor-Corrector samplers with customizations for inpainting."""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/sampling/pc_sampler.py

from __future__ import annotations

from typing import Tuple, TypeVar

import torch
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
)
from mattergen.diffusion.sampling.pc_sampler import (
    _mask_replace,
)
from tqdm.auto import tqdm

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time

SampleAndMeanAndMaybeRecords = Tuple[
    Diffusable, Diffusable, list[Diffusable] | None
]


class AdditionalDataPredictorCorrector(GuidedPredictorCorrector):
    """Predictor-Corrector sampler that handles additional data."""

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        recorded_means = None
        if record:
            recorded_samples = []
            recorded_means = []
            recorded_scores = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(
            self._max_t, self._eps_t, self.N, device=self._device
        )
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(
            self._device
        )

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full(
                (batch.get_batch_size(),), timesteps[i], device=self._device
            )

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score
                        for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[
                        str, Tuple[torch.Tensor, torch.Tensor]
                    ] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(
                            batch
                        ),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                        recorded_means.append(mean_batch.clone().to("cpu"))
                        recorded_scores.append(score["pos"].clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=mask,
                    )

            # Predictor updates
            score = self._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score
                for k, predictor in self._predictors.items()
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
                recorded_means.append(mean_batch.clone().to("cpu"))
                recorded_scores.append(score["pos"].clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=mask,
            )

        if record:
            recorded_scores = torch.stack(recorded_scores, dim=0)

            # optionally, save to disk
            torch.save(recorded_scores, "recorded_scores.pt")
            torch.save(
                batch["atomic_numbers"].clone().to("cpu"),
                "recorded_atomic_numbers.pt",
            )
            torch.save(
                batch.get_batch_idx("pos").clone().to("cpu"),
                "recorded_batch_idx.pt",
            )

        return batch, mean_batch, recorded_samples, recorded_means


class CustomGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Predictor-Corrector adding noise to masked regions."""

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        dummy_mask = {}
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
            dummy_mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
            dummy_mask.setdefault(k, None)
        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(
            self._max_t, self._eps_t, self.N, device=self._device
        )
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(
            self._device
        )
        batch0 = batch.clone()

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full(
                (batch.get_batch_size(),), timesteps[i], device=self._device
            )

            noisy_sample = self.diffusion_module.corruption.sample_marginal(
                batch0, t
            )

            batch["pos"] = batch["pos"].lerp_(noisy_sample["pos"], mask["pos"])

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score
                        for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[
                        str, Tuple[torch.Tensor, torch.Tensor]
                    ] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(
                            batch
                        ),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=dummy_mask,
                    )

            # Predictor updates
            score = self._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score
                for k, predictor in self._predictors.items()
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
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=dummy_mask,
            )

        return batch, mean_batch, recorded_samples


class RePaintLegacyGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Predictor-Corrector adding noise to masked regions following RePaint.

    This is a legacy class kept for reference.
    """

    def __init__(self, **kwargs):
        """Initialize RePaintLegacyGuidedPredictorCorrector."""
        self.n_resample_steps = kwargs.pop("n_resample_steps", 1)

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
        timesteps = torch.linspace(
            self._max_t, self._eps_t, self.N, device=self._device
        )
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(
            self._device
        )

        batch0 = batch.clone()
        noisy_sample = self.diffusion_module.corruption.sample_marginal(
            batch0,
            torch.full(
                (batch.get_batch_size(),), timesteps[0], device=self._device
            ),
        )
        batch["pos"] = batch["pos"].lerp_(noisy_sample["pos"], mask["pos"])

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full(
                (batch.get_batch_size(),), timesteps[i], device=self._device
            )

            for i_res in range(self.n_resample_steps):
                # Predictor updates
                score = self._score_fn(batch, t)
                predictor_fns = {
                    k: predictor.update_given_score
                    for k, predictor in self._predictors.items()
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

                # TR: I set mask to None, so that the predictor steps are
                # also passed to the corrector
                batch, mean_batch = _mask_replace(
                    samples_means=samples_means,
                    batch=batch,
                    mean_batch=mean_batch,
                    mask=ignore_mask,
                )

                # Corrector updates.
                if self._correctors:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t + dt)
                        fns = {
                            k: corrector.step_given_score
                            for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[
                            str, Tuple[torch.Tensor, torch.Tensor]
                        ] = apply(
                            fns=fns,
                            broadcast={"t": t + dt, "dt": dt},
                            x=batch,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(
                                batch
                            ),
                        )
                        if record:
                            recorded_samples.append(batch.clone().to("cpu"))
                        # TR: Setting mask to None with the same reason as
                        # mentioned above
                        batch, mean_batch = _mask_replace(
                            samples_means=samples_means,
                            batch=batch,
                            mean_batch=mean_batch,
                            mask=ignore_mask,
                        )
                if i < self.N - 1 and True:
                    noisy_sample = (
                        self.diffusion_module.corruption.sample_marginal(
                            batch0, t + dt if i < self.N - 1 else t
                        )
                    )
                    batch["pos"] = batch["pos"].lerp_(
                        noisy_sample["pos"], mask["pos"]
                    )

                if i_res < self.n_resample_steps - 1 and i < self.N - 1:
                    # Get the sqrt(sigma_{t-1} ** 2 - sigma_{t-2} ** 2)
                    only_for_z, sigma_t_prev_t_prev2, _ = self._predictors[
                        "pos"
                    ]._get_coeffs(
                        x=batch["pos"],
                        t=t + dt,
                        batch=batch,
                        dt=dt,
                        batch_idx=self._multi_corruption._get_batch_indices(
                            batch
                        )["pos"],
                    )
                    z = torch.randn_like(only_for_z)
                    batch["pos"] = self._multi_corruption.corruptions[
                        "pos"
                    ].wrap(batch["pos"] + sigma_t_prev_t_prev2 * z)

        return batch, mean_batch, recorded_samples


def time_jump_scheduler(t_T=250, jump_len=10, jump_n_sample=10):
    """Create a time schedule with jumps for RePaint-like sampling."""
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1

    # Start at t_T
    t = t_T
    ts = []

    # Continue until t drops below 1
    while t >= 1:
        # Decrement t by 1
        t -= 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t += 1
                ts.append(t)
    # ts.append(-1)
    return ts


class RePaintV2GuidedPredictorCorrector(GuidedPredictorCorrector):
    """Predictor-Corrector adding noise to masked regions following RePaint."""

    def __init__(self, **kwargs):
        """Initialize RePaintV2GuidedPredictorCorrector."""
        self.n_resample_steps = kwargs.pop("n_resample_steps", 1)
        self.jump_length = kwargs.pop("jump_length", 10)

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
        recorded_means = None
        ignore_mask = {}
        if record:
            recorded_samples = []
            recorded_means = []
        for k in self._predictors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        mean_batch = batch.clone()

        # ignore_mask = mask

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(
            self._eps_t, self._max_t, self.N, device=self._device
        )
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(
            self._device
        )

        batch0 = batch.clone()

        scheduled_timesteps = time_jump_scheduler(
            t_T=self.N,
            jump_len=self.jump_length,
            jump_n_sample=self.n_resample_steps,
        )

        t_last = 1000
        for i in tqdm(scheduled_timesteps, miniters=50, mininterval=5):
            # Set the timestep
            t_cur = timesteps[i]
            t = torch.full(
                (batch.get_batch_size(),), t_cur, device=self._device
            )

            if t_cur < t_last:
                noisy_sample = (
                    self.diffusion_module.corruption.sample_marginal(batch0, t)
                )
                batch["pos"] = batch["pos"].lerp_(
                    noisy_sample["pos"], mask["pos"]
                )

                # for i_res in range(self.n_resample_steps):
                # Predictor updates
                score = self._score_fn(batch, t)
                predictor_fns = {
                    k: predictor.update_given_score
                    for k, predictor in self._predictors.items()
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
                    recorded_means.append(mean_batch.clone().to("cpu"))

                # TR: I set mask to None, so that the predictor steps are
                # also passed to the corrector
                batch, mean_batch = _mask_replace(
                    samples_means=samples_means,
                    batch=batch,
                    mean_batch=mean_batch,
                    mask=ignore_mask,
                )

                # Corrector updates.
                if self._correctors and i > 0:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t + dt)
                        fns = {
                            k: corrector.step_given_score
                            for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[
                            str, Tuple[torch.Tensor, torch.Tensor]
                        ] = apply(
                            fns=fns,
                            broadcast={"t": t + dt, "dt": dt},
                            x=batch,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(
                                batch
                            ),
                        )
                        if record:
                            recorded_samples.append(batch.clone().to("cpu"))
                            recorded_means.append(mean_batch.clone().to("cpu"))
                        # TR: Setting mask to None with the same
                        # reason as mentioned above
                        batch, mean_batch = _mask_replace(
                            samples_means=samples_means,
                            batch=batch,
                            mean_batch=mean_batch,
                            mask=ignore_mask,
                        )
            else:
                t = torch.full(
                    (batch.get_batch_size(),), t_cur, device=self._device
                )
                if i > 0:
                    only_for_z, sigma_t_prev_t_prev2, _ = self._predictors[
                        "pos"
                    ]._get_coeffs(
                        x=batch["pos"],
                        t=t + dt,
                        batch=batch,
                        dt=dt,
                        batch_idx=self._multi_corruption._get_batch_indices(
                            batch
                        )["pos"],
                    )
                    z = torch.randn_like(only_for_z)
                    batch["pos"] = self._multi_corruption.corruptions[
                        "pos"
                    ].wrap(batch["pos"] + sigma_t_prev_t_prev2 * z)

            t_last = t_cur

        return batch, mean_batch, recorded_samples, recorded_means


class TDPaintGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Predictor-Corrector following TDPaint."""

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        recorded_means = None
        ignore_mask = {}
        if record:
            recorded_samples = []
            recorded_means = []
            recorded_scores = []
        for k in self._predictors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
            ignore_mask.setdefault(k, None)
        mean_batch = batch.clone()

        ignore_mask = mask

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(
            self._max_t, self._eps_t, self.N, device=self._device
        )
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(
            self._device
        )

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            batch_size = batch.get_batch_size()
            batch_size = batch["pos"].shape[0]
            t = torch.full((batch_size,), timesteps[i], device=self._device)

            t[mask["pos"][:, 0].bool()] = self._eps_t

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score
                        for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[
                        str, Tuple[torch.Tensor, torch.Tensor]
                    ] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        # batch_idx={'pos': None},#
                        batch_idx=self._multi_corruption._get_batch_indices(
                            batch
                        ),
                        mask={"pos": mask["pos"][:, 0]},
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                        recorded_means.append(mean_batch.clone().to("cpu"))
                        recorded_scores.append(score["pos"].clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=ignore_mask,
                    )

            # Predictor updates
            score = self._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score
                for k, predictor in self._predictors.items()
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
                recorded_means.append(mean_batch.clone().to("cpu"))
                recorded_scores.append(score["pos"].clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=ignore_mask,
            )

        if record:
            recorded_scores = torch.stack(recorded_scores, dim=0)

            # optionally, save to disk
            torch.save(recorded_scores, "recorded_scores.pt")
            torch.save(
                batch["atomic_numbers"].clone().to("cpu"),
                "recorded_atomic_numbers.pt",
            )
            torch.save(
                batch.get_batch_idx("pos").clone().to("cpu"),
                "recorded_batch_idx.pt",
            )

        return batch, mean_batch, recorded_samples, recorded_means
