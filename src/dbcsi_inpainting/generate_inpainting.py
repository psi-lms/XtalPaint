from torch.utils.data import DataLoader
import torch
from pymatgen.core.structure import Structure
from typing import Literal
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pymatgen.core import Structure
import os

from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.types import TargetProperty
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo, PRETRAINED_MODEL_NAME
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


class CrystalGeneratorInpainting(CrystalGenerator):
    
    dataloader: DataLoader
    
    def __init__(self, dataloader: DataLoader, *args, **kwargs):
        self.dataloader = dataloader
        super().__init__(*args, **kwargs)
        
    def check_distributed(self, sampler):
        n_cuda_devices = torch.cuda.device_count()
        if n_cuda_devices:
            print(f'Found {n_cuda_devices} cuda devices')
            model = sampler.diffusion_module.model
            model = torch.nn.DataParallel(model)
            model.to(torch.device("cuda:0"))
            sampler.diffusion_module.model = model 
            
    
    def get_condition_loader(self, *args, **kwargs):
        return self.dataloader

    def generate(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        target_compositions_dict: list[dict[str, float]] | None = None,
        output_dir: str = "outputs",
    ) -> list[Structure]:
        # Prioritize the runtime provided batch_size, num_batches and target_compositions_dict
        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        target_compositions_dict = target_compositions_dict or self.target_compositions_dict
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
        condition_loader = self.get_condition_loader(sampling_config, target_compositions_dict)

        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(pl_module=self.model)
        
        # self.check_distributed(sampler=sampler)
        
        sampler.diffusion_module.model.denoise_atom_types = False
        sampler._multi_corruption.corruptions.pop('atomic_numbers', None)

        print(sampler.diffusion_module.corruption.corruptions)

        print(sampler._predictors.keys())
        print(sampler._correctors.keys())
        print(sampler._n_steps_corrector)

        generated_structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            cfg=self.cfg,
            output_path=Path(output_dir),
            properties_to_condition_on=self.properties_to_condition_on,
            record_trajectories=self.record_trajectories,
        )

        return generated_structures

def generate_reconstructed_structures(
    structures_to_reconstruct: DataLoader,
    pretrained_name: PRETRAINED_MODEL_NAME | None = 'mattergen_base',
    output_path: str = None,
    model_path: str = None,     # '/data/user/reents_t/projects/mlip/git/mattergen/checkpoints/mattergen_base',
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
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None

    generator = CrystalGeneratorInpainting(
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
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
    )
    
    return generator.generate(output_dir=Path(output_path))
