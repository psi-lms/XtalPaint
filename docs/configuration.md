# Workflow Configuration

XtalPaint uses a single configuration object — `XtalPaintConfig` — to drive both plain-Python and AiiDA-based workflows. This page explains how to build that config, what each field does, and how the same config object is shared between the two execution modes.

---

## Design principles

- **Presence = enabled, `None` = skip.** Each pipeline stage is controlled by its own typed config object. If the field is `None` the stage is omitted — no boolean flags needed.
- **AiiDA options are isolated.** Everything in the optional `aiida` block (code labels, scheduler resources) is invisible to plain-Python execution.
- **Flat, validated inputs.** Pydantic validates every field at construction time so mistakes surface immediately rather than at run time.

---

## Top-level structure

```python
from xtalpaint.inpainting.config_schema import (
    XtalPaintConfig,
    CandidateGenerationConfig,
    InpaintingConfig,
    RefinementConfig,
    RelaxationConfig,
    RelaxationParams,
    UniquenessConfig,
    AiiDAOptions,
    AiiDATaskOptions,
)

config = XtalPaintConfig(
    structures=...,                    # required — dict[str, Structure] or BatchedStructures
    run_inpainting=True,               # set False to skip the diffusion step
    candidate_generation=...,          # CandidateGenerationConfig | None
    pre_refinement=...,                # RefinementConfig | None
    inpainting=...,                    # InpaintingConfig  (always required)
    relaxation=...,                    # RelaxationConfig | None
    aiida=...,                         # AiiDAOptions | None  (ignored outside AiiDA)
)
```

The pipeline runs in this order when a stage is enabled:

```
candidate_generation → inpainting → pre_refinement → relaxation
                                                        ├─ constrained pass
                                                        ├─ full pass (on constrained output)
                                                        └─ full_direct pass (on inpainted directly)
                                                      each pass: → [refine] → [filter_unique]
```

---

## Stage reference

### Input structures

```python
from pymatgen.core import Structure

config = XtalPaintConfig(
    structures={"host_001": Structure(...), "host_002": Structure(...)},
    inpainting=...,
)
```

`structures` accepts:

- `dict[str, Structure]` — plain pymatgen structures
- `BatchedStructures` — XtalPaint's batched wrapper
- AiiDA `StructureData` / `InpaintingStructureData` — when running inside AiiDA

---

### Candidate generation

Required when the input structures are plain `Structure` objects (not yet marked as inpainting targets). Omit this block if your structures are already `InpaintingStructureData` instances.

```python
candidate_generation=CandidateGenerationConfig(
    n_inp=2,                  # int or dict[str, int] — number of sites to inpaint
    element="H",              # element to place; dict[str, str] for per-structure control
    num_samples=1,            # how many candidate sets to generate
)
```

For per-structure control over the number of sites and element:

```python
candidate_generation=CandidateGenerationConfig(
    n_inp={"host_001": 2, "host_002": 4},
    element={"host_001": "H", "host_002": "Li"},
)
```

---

### Inpainting

The core diffusion stage. All sampling parameters live in one flat block.

```python
inpainting=InpaintingConfig(
    # Model — provide exactly one of these:
    pretrained_name="mattergen_base",   # use a bundled pretrained checkpoint
    # model_path="/path/to/checkpoint", # or point to a local file

    # Sampling
    predictor_corrector="baseline",     # see supported keys below
    N_steps=5,
    coordinates_snr=0.2,
    n_corrector_steps=1,
    batch_size=1000,

    # Optional
    fix_cell=True,                      # keep unit cell fixed during sampling
    record_trajectories=False,
    sampling_config_path=None,          # override MatterGen sampling config dir
)
```

**Supported `predictor_corrector` values:**

| Key | Description |
|---|---|
| `baseline` | Standard guided predictor-corrector |
| `baseline-with-noise` | Custom variant with additional noise |
| `baseline-store-scores` | Records score function outputs |
| `repaint-v1` | RePaint resampling (legacy) |
| `repaint-v2` | RePaint resampling (v2) |
| `TD` | Time-dependent (TD-Paint) variant |

!!! note "Repaint variants"
    When using `repaint-v1` or `repaint-v2`, you must also set `n_resample_steps` and `jump_length`:

    ```python
    inpainting=InpaintingConfig(
        predictor_corrector="repaint-v2",
        n_resample_steps=10,
        jump_length=5,
        # ... other fields ...
    )
    ```

---

### Pre-refinement

Optional symmetry refinement applied *after* inpainting and *before* relaxation.

```python
pre_refinement=RefinementConfig(
    symprec=0.01,      # symmetry precision for SpacegroupAnalyzer
    primitive=False,   # if True, convert to primitive cell
)
```

Omit `pre_refinement` (or set it to `None`) to skip this step.

---

### Relaxation

The relaxation stage is split into two distinct layers:

- **`RelaxationParams`** — the inputs forwarded directly to `relax_structures()` (MLIP, optimiser, convergence)
- **`RelaxationConfig`** — workflow-level controls: which passes to run and post-relaxation processing

#### Relaxation passes

Three passes can be run independently or in combination:

| Flag | Behaviour | WorkGraph label |
|---|---|---|
| `constrained=True` *(default)* | Relax only `elements_to_relax` | `inpainted_constrained_relaxation` |
| `full=True` | Full relax on the *constrained* output | `pre_relaxed_inpainted_full_relaxation` |
| `full_direct=True` | Full relax directly on inpainted structures | `unrelaxed_inpainted_full_relaxation` |

`full` and `full_direct` together give a direct comparison between relaxing from the raw inpainted geometry versus relaxing from an already-constrained geometry.

#### Post-relaxation steps

`refine` and `filter_unique` run *after each active pass*, in order: relax → refine → deduplicate.

```python
relaxation=RelaxationConfig(
    params=RelaxationParams(
        mlip="mattersim",
        optimizer="BFGS",
        fmax=0.05,
        max_n_steps=500,
        elements_to_relax=["H"],       # required when constrained=True
        return_initial_energies=False,
        return_final_forces=False,
    ),
    # Which passes to run:
    constrained=True,      # relax only H atoms
    full=True,             # then do a full relax on that output
    full_direct=False,     # skip direct full relax

    # Post-relaxation processing (applied to each pass):
    refine=True,
    refinement_symprec=0.01,
    refinement_primitive=False,
    filter_unique=True,
    uniqueness=UniquenessConfig(
        symprec=0.01,
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
    ),
)
```

!!! warning "Constraints for `constrained`"
    `constrained=True` requires `params.elements_to_relax` to be set.
    `full=True` requires `constrained=True` (the full-relax pass operates on the constrained output).

---

## Running without AiiDA

Without AiiDA, pass the `inpainting` config directly to the pipeline functions. The `aiida` block is simply not set.

```python
from xtalpaint.inpainting.config_schema import XtalPaintConfig, InpaintingConfig
from xtalpaint.inpainting.inpainting_process import run_inpainting_pipeline
from xtalpaint.utils.relaxation_utils import relax_structures

config = XtalPaintConfig(
    structures={"host_001": structure},
    candidate_generation=CandidateGenerationConfig(n_inp=2, element="H"),
    inpainting=InpaintingConfig(
        model_path="/path/to/checkpoint.ckpt",
        predictor_corrector="baseline",
        N_steps=5,
        coordinates_snr=0.2,
        n_corrector_steps=1,
        batch_size=1000,
        sampling_config_path="/path/to/sampling_conf",
    ),
)

# Run inpainting
results = run_inpainting_pipeline(
    structures=config.structures,
    config=config.inpainting,          # pass InpaintingConfig directly
)
inpainted = results["structures"]

# Optional relaxation (using config.relaxation.relax_inputs())
if config.relaxation is not None:
    relaxed = relax_structures(
        inpainted,
        **config.relaxation.relax_inputs(constrained=True),
    )
```

!!! tip
    `InpaintingConfig.model_dump(exclude_none=True)` produces a plain dict that the pipeline functions also accept, which is convenient when serialising configs to JSON/YAML.

---

## Running with AiiDA

Add the `aiida` block to the same `XtalPaintConfig`. Everything else stays identical — the pipeline stages, their parameters, and all validation remain unchanged.

```python
from xtalpaint.inpainting.config_schema import AiiDAOptions, AiiDATaskOptions
from xtalpaint.aiida.workgraphs.inpainting_graph_task import setup_inpainting_wg

config = XtalPaintConfig(
    structures=...,
    candidate_generation=...,
    inpainting=...,
    relaxation=...,

    # AiiDA-specific block — ignored in plain-Python runs
    aiida=AiiDAOptions(
        default_code_label="xtalpaint@localhost",   # fallback for all tasks
        relax_code_label="xtalpaint@hpc",           # override for relaxation
        inpainting_options=AiiDATaskOptions(
            resources={"num_machines": 1, "num_mpiprocs_per_machine": 4},
            max_wallclock_seconds=3600,
            withmpi=True,
        ),
        relax_options=AiiDATaskOptions(
            resources={"num_machines": 2, "num_mpiprocs_per_machine": 8},
            withmpi=True,
        ),
    ),
)

# Build and submit the WorkGraph
wg = setup_inpainting_wg(config)
wg.submit()
```

### Code label resolution

Each task resolves its code label in order:

1. Task-specific label (`inpainting_code_label`, `relax_code_label`, `candidate_generation_code_label`)
2. Fall back to `default_code_label`

This lets you run most tasks on one machine and override just the resource-heavy ones.

### `AiiDATaskOptions` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `resources` | `dict` | `{}` | AiiDA scheduler resource dict |
| `max_wallclock_seconds` | `int \| None` | `None` | Wall-clock limit |
| `queue_name` | `str \| None` | `None` | Scheduler queue/partition |
| `withmpi` | `bool` | `False` | Enable MPI-parallel execution |

---

## Full examples

=== "Without AiiDA"

    ```python
    from pymatgen.core import Structure
    from xtalpaint.inpainting.config_schema import (
        XtalPaintConfig,
        CandidateGenerationConfig,
        InpaintingConfig,
        RefinementConfig,
        RelaxationConfig,
        RelaxationParams,
    )
    from xtalpaint.inpainting.inpainting_process import run_inpainting_pipeline

    structure = Structure.from_file("host.cif")

    config = XtalPaintConfig(
        structures={"host": structure},
        candidate_generation=CandidateGenerationConfig(
            n_inp=2,
            element="H",
        ),
        inpainting=InpaintingConfig(
            pretrained_name="mattergen_base",
            predictor_corrector="baseline",
            N_steps=5,
            coordinates_snr=0.2,
            n_corrector_steps=1,
            batch_size=1000,
        ),
        pre_refinement=RefinementConfig(symprec=0.01),
        relaxation=RelaxationConfig(
            params=RelaxationParams(
                mlip="mattersim",
                optimizer="BFGS",
                elements_to_relax=["H"],
                fmax=0.05,
            ),
            constrained=True,
            refine=True,
            filter_unique=True,
        ),
        # no aiida block → plain Python execution
    )

    results = run_inpainting_pipeline(
        structures=config.structures,
        config=config.inpainting,
    )
    print(results["structures"])
    ```

=== "With AiiDA"

    ```python
    from pymatgen.core import Structure
    from xtalpaint.inpainting.config_schema import (
        XtalPaintConfig,
        CandidateGenerationConfig,
        InpaintingConfig,
        RefinementConfig,
        RelaxationConfig,
        RelaxationParams,
        AiiDAOptions,
        AiiDATaskOptions,
    )
    from xtalpaint.aiida.workgraphs.inpainting_graph_task import setup_inpainting_wg

    structure = Structure.from_file("host.cif")

    config = XtalPaintConfig(
        structures={"host": structure},
        candidate_generation=CandidateGenerationConfig(
            n_inp=2,
            element="H",
        ),
        inpainting=InpaintingConfig(
            pretrained_name="mattergen_base",
            predictor_corrector="baseline",
            N_steps=5,
            coordinates_snr=0.2,
            n_corrector_steps=1,
            batch_size=1000,
        ),
        pre_refinement=RefinementConfig(symprec=0.01),
        relaxation=RelaxationConfig(
            params=RelaxationParams(
                mlip="mattersim",
                optimizer="BFGS",
                elements_to_relax=["H"],
                fmax=0.05,
            ),
            constrained=True,
            refine=True,
            filter_unique=True,
        ),
        aiida=AiiDAOptions(
            default_code_label="xtalpaint@localhost",
            relax_code_label="xtalpaint@hpc",
            inpainting_options=AiiDATaskOptions(
                resources={"num_machines": 1, "num_mpiprocs_per_machine": 4},
                withmpi=True,
            ),
            relax_options=AiiDATaskOptions(
                resources={"num_machines": 2, "num_mpiprocs_per_machine": 8},
                withmpi=True,
            ),
        ),
    )

    wg = setup_inpainting_wg(config)
    wg.submit()
    ```

The two snippets are identical except for the `aiida=` block. This means you can develop and test workflows locally (without AiiDA) and then promote them to a remote HPC environment by adding the `aiida` block — no other changes needed.

---

## Configuration reference summary

| Class | Required fields | Purpose |
|---|---|---|
| `XtalPaintConfig` | `structures`, `inpainting` | Top-level workflow config |
| `CandidateGenerationConfig` | `n_inp`, `element` | Generate inpainting masks |
| `InpaintingConfig` | `predictor_corrector`, `N_steps`, `coordinates_snr`, `n_corrector_steps`, `batch_size`, one of `pretrained_name`/`model_path` | Diffusion sampling |
| `RefinementConfig` | — | Symmetry refinement before relaxation |
| `RelaxationGraphConfig` | `params` | Single-pass input for `relaxation_graph` |
| `RelaxationConfig` | `params` | Multi-pass relaxation stage (extends `RelaxationGraphConfig`) |
| `RelaxationParams` | `mlip`, `optimizer` | Inputs forwarded to `relax_structures()` |
| `UniquenessConfig` | — | Deduplication tolerances |
| `AiiDAOptions` | — | Code labels + per-task scheduler options |
| `AiiDATaskOptions` | — | Resources, wall-clock, MPI flag |

### Using `relaxation_graph` directly

`relaxation_graph` accepts `RelaxationGraphConfig` directly, so you can call it outside the inpainting WorkGraph without needing the full `RelaxationConfig` (which carries inpainting-WG-specific pass flags):

```python
from xtalpaint.aiida.workgraphs.relaxation import relaxation_graph
from xtalpaint.inpainting.config_schema import (
    RelaxationGraphConfig, RelaxationParams, UniquenessConfig,
    AiiDATaskOptions,
)

relax_cfg = RelaxationGraphConfig(
    params=RelaxationParams(mlip="mattersim", optimizer="BFGS", elements_to_relax=["H"]),
    refine=True,
    filter_unique=True,
    uniqueness=UniquenessConfig(symprec=0.01),
)

out = relaxation_graph(
    structures=my_structures,
    relax_config=relax_cfg,
    aiida_options=AiiDATaskOptions(withmpi=True, resources={"num_machines": 1}),
    code_label="xtalpaint@hpc",
    constrained=True,          # True → include elements_to_relax; False → full relax
)
```

Since `RelaxationConfig` inherits from `RelaxationGraphConfig`, you can also pass a `RelaxationConfig` directly wherever `RelaxationGraphConfig` is expected.
