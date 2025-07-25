from torch.utils.data import DataLoader
import torch
from pymatgen.core.structure import Structure
from typing import Literal
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pymatgen.core.trajectory import Trajectory


from mattergen.common.data.types import TargetProperty
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import (
    MatterGenCheckpointInfo,
    PRETRAINED_MODEL_NAME,
)
from mattergen.generator import (
    CrystalGenerator,
    list_of_time_steps_to_list_of_trajectories,
    structure_from_model_output,
    structures_from_trajectory,
)

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure
from tqdm import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.condition_factory import ConditionLoader

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import (
    MatterGenCheckpointInfo,
    save_structures,
)

from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector


def draw_samples_from_sampler(
    sampler: PredictorCorrector,
    condition_loader: ConditionLoader,
    properties_to_condition_on: TargetProperty | None = None,
    record_trajectories: bool = True,
    fix_cell: bool = True,
) -> list[Structure]:

    # Dict
    properties_to_condition_on = properties_to_condition_on or {}

    # we cannot conditional sample on something on which the model was not trained to condition on
    assert all([key in sampler.diffusion_module.model.cond_fields_model_was_trained_on for key in properties_to_condition_on.keys()])  # type: ignore

    all_samples_list = []
    all_trajs_list = []
    for conditioning_data, mask in tqdm(
        condition_loader, desc="Generating samples"
    ):

        # generate samples
        if record_trajectories:
            # sample, mean, intermediate_samples = sampler.sample_with_record(conditioning_data, mask)
            # sample, mean, intermediate_samples, intermediate_means = sampler.sample_with_record(conditioning_data, mask)
            _out = sampler.sample_with_record(conditioning_data, mask)
            if len(_out) == 4:
                sample, mean, intermediate_samples, intermediate_means = _out
            elif len(_out) == 3:
                sample, mean, intermediate_samples = _out
            all_trajs_list.extend(
                list_of_time_steps_to_list_of_trajectories(
                    intermediate_samples
                )
            )
        else:
            sample, mean = sampler.sample(conditioning_data, mask)
        all_samples_list.extend(mean.to_data_list())
    all_samples = collate(all_samples_list)
    assert isinstance(all_samples, ChemGraph)
    lengths, angles = lattice_matrix_to_params_torch(all_samples.cell)
    all_samples = all_samples.replace(lengths=lengths, angles=angles)

    generated_strucs = structure_from_model_output(
        all_samples["pos"].reshape(-1, 3),
        all_samples["atomic_numbers"].reshape(-1),
        all_samples["lengths"].reshape(-1, 3),
        all_samples["angles"].reshape(-1, 3),
        all_samples["num_atoms"].reshape(-1),
    )

    trajectories = []
    for ix, traj in enumerate(all_trajs_list):
        strucs = structures_from_trajectory(traj)
        trajectories.append(
            Trajectory.from_structures(
                strucs,
                constant_lattice=fix_cell,
            )
        )

    return generated_strucs, trajectories


class CrystalInpaintingGenerator(CrystalGenerator):

    dataloader: DataLoader

    def __init__(self, dataloader: DataLoader, *args, **kwargs):
        self.dataloader = dataloader
        super().__init__(*args, **kwargs)

    def get_condition_loader(self, *args, **kwargs):
        return self.dataloader

    def generate(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        target_compositions_dict: list[dict[str, float]] | None = None,
        fix_cell: bool = True,
    ) -> list[Structure]:
        # Prioritize the runtime provided batch_size, num_batches and target_compositions_dict
        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        target_compositions_dict = (
            target_compositions_dict or self.target_compositions_dict
        )
        assert batch_size is not None
        assert num_batches is not None

        # print config for debugging and reproducibility
        print("\nModel config:")
        print(OmegaConf.to_yaml(self.cfg, resolve=True))

        sampling_config = self.load_sampling_config(
            batch_size=batch_size,
            num_batches=num_batches,
            target_compositions_dict=target_compositions_dict,
        )

        print("\nSampling config:")
        print(OmegaConf.to_yaml(sampling_config, resolve=True))
        condition_loader = self.get_condition_loader(
            sampling_config, target_compositions_dict
        )

        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(pl_module=self.model)

        sampler.diffusion_module.model.denoise_atom_types = False
        sampler._multi_corruption.corruptions.pop("atomic_numbers", None)

        print(sampler.diffusion_module.corruption.corruptions)

        generated_structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            properties_to_condition_on=self.properties_to_condition_on,
            record_trajectories=self.record_trajectories,
            fix_cell=fix_cell,
        )

        return generated_structures


def generate_reconstructed_structures(
    structures_to_reconstruct: DataLoader,
    pretrained_name: PRETRAINED_MODEL_NAME | None = "mattergen_base",
    model_path: str = None,  # '/data/user/reents_t/projects/mlip/git/mattergen/checkpoints/mattergen_base',
    batch_size: int = 10,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
    fix_cell: bool = True,
):
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default: default)
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.
        target_compositions: List of dictionaries with target compositions to condition on. Each dictionary should have the form `{element: number_of_atoms}`. If None, the target compositions are not conditioned on.
           Only supported for models trained for crystal structure prediction (CSP) (default: None)

    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."
    assert (
        pretrained_name is None or model_path is None
    ), "Only one of pretrained_name or model_path can be provided."

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    if pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            pretrained_name, config_overrides=config_overrides
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading,
        )
    _sampling_config_path = (
        Path(sampling_config_path)
        if sampling_config_path is not None
        else None
    )

    generator = CrystalInpaintingGenerator(
        dataloader=structures_to_reconstruct,
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor
            if diffusion_guidance_factor is not None
            else 0.0
        ),
        target_compositions_dict=target_compositions,
    )

    return generator.generate(fix_cell=fix_cell)
