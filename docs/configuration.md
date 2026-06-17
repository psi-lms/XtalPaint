# Workflow Configuration

XtalPaint uses a single configuration object — `XtalPaintConfig` — to drive both plain-Python and AiiDA-based workflows. This page explains how to build that config, what each field does, and how the same config object is shared between the two execution modes.

---

## Design principles

- **Structures are not part of the config.** The input structures are passed separately — to the pipeline functions or to `InpaintingWorkGraph.build()` — so the same config can be reused across different structure sets.
- **Presence = enabled, `None` = skip — for pipeline stages.** Each pipeline stage (`candidate_generation`, `pre_refinement`, `relaxation`) is controlled by its own typed config object; if the field is `None` the stage is omitted. The post-relaxation steps *inside* the relaxation graph (`refinement`, `uniqueness`) are instead toggled with their `include_task` flag.
- **AiiDA options are co-located with their stage.** Relaxation AiiDA settings live inside `RelaxationGraphConfig.aiida`; pipeline-level AiiDA settings (inpainting, candidate generation, pre-refinement) live in `XtalPaintConfig.aiida`.
- **Flat, validated inputs.** Pydantic validates every field at construction time so mistakes surface immediately rather than at run time.

---

## Top-level structure

```python
from xtalpaint.inpainting.config_schema import (
    XtalPaintConfig,
    CandidateGenerationConfig,
    InpaintingConfig,
    RefinementConfig,
    InpaintingRelaxationConfig,
    RelaxationGraphConfig,
    RelaxationParams,
    UniquenessConfig,
    RelaxationAiiDAOptions,
    AiiDAOptions,
    AiiDATaskOptions,
)

config = XtalPaintConfig(
    run_inpainting=True,               # set False to skip candidate generation + diffusion
    candidate_generation=...,          # CandidateGenerationConfig | None
    pre_refinement=...,                # RefinementConfig | None
    inpainting=...,                    # InpaintingConfig  (always required)
    relaxation=...,                    # InpaintingRelaxationConfig | None
    aiida=...,                         # AiiDAOptions | None  (ignored outside AiiDA)
)
```

The pipeline runs in this order when a stage is enabled:

```text
candidate_generation → inpainting → pre_refinement → relaxation
                                                        ├─ constrained pass
                                                        ├─ full pass (on constrained output)
                                                        └─ full_direct pass (on inpainted directly)
                                                      each pass: → [refinement] → [uniqueness]
```

---

## Stage reference

### Input structures

Structures are passed alongside the config, not inside it:

```python
from pymatgen.core import Structure
from xtalpaint.data import BatchedStructures

structures = BatchedStructures(
    {"host_001": Structure(...), "host_002": Structure(...)}
)

# Plain Python
results = run_inpainting_pipeline(structures=structures, config=config.inpainting)

# AiiDA
wg = InpaintingWorkGraph.build(structures=structures, inputs=config)
```

Accepted structure inputs:

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
    pretrained_name="TD-pos-only",      # XtalPaint model (auto-downloaded) or a MatterGen checkpoint
    # model_path="/path/to/model_dir",  # or point to a local checkpoint directory

    # Sampling
    predictor_corrector="TD",           # see supported combinations below
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

!!! tip "Recommended model: `TD-pos-only`"
    `pretrained_name` accepts the XtalPaint models hosted on
    [Hugging Face](https://huggingface.co/t-reents/XtalPaint) (`TD-pos-only`,
    `pos-only`) as well as MatterGen's own checkpoints. The XtalPaint models are
    downloaded automatically and cached the first time they are selected. For
    accurate inpainting we **recommend the `TD-pos-only` model** — the core
    model of XtalPaint — with the time-dependent (`TD`) predictor-corrector.

    Alternatively, download a checkpoint explicitly and pass it as `model_path`:

    ```python
    from xtalpaint.models import download_pretrained_model

    model_path = download_pretrained_model("TD-pos-only")
    ```

    Selecting `predictor_corrector="TD"` without the `TD-pos-only` model (or
    pointing `model_path` at a checkpoint that has not been downloaded) raises
    an error with these instructions.

**Supported `predictor_corrector` values** — the `TD` variant requires the `TD-pos-only` model; the others are used with the `pos-only` model or a MatterGen checkpoint:

| Key | Description |
|---|---|
| `baseline` | Standard guided predictor-corrector |
| `baseline-with-noise` | Custom variant with additional noise |
| `baseline-store-scores` | Records score function outputs |
| `repaint-v1` | RePaint resampling (legacy) |
| `repaint-v2` | RePaint resampling (v2) |
| `TD` | Time-dependent (TD-Paint) variant — requires the `TD-pos-only` model |

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

Omit `pre_refinement` (or set it to `None`) to skip this step. Unlike the post-relaxation refinement below, this stage is enabled purely by presence — its `include_task` flag is ignored.

---

### Relaxation

The relaxation stage is split into three layers:

- **`RelaxationParams`** — the inputs forwarded directly to `relax_structures()` (MLIP, optimiser, convergence)
- **`RelaxationGraphConfig`** — one relaxation pass: `params` plus the optional post-relaxation steps (`refinement`, `uniqueness`) and the AiiDA options (`aiida`)
- **`InpaintingRelaxationConfig`** — workflow-level orchestration: which passes to run (`constrained` / `full` / `full_direct`), wrapping a shared `relax_config`

#### Relaxation passes

Three passes can be run independently or in combination:

| Flag | Behaviour | WorkGraph label |
|---|---|---|
| `constrained=True` *(default)* | Relax only `elements_to_relax` | `inpainted_constrained_relaxation` |
| `full=True` | Full relax on the *constrained* output | `pre_relaxed_inpainted_full_relaxation` |
| `full_direct=True` | Full relax directly on inpainted structures | `unrelaxed_inpainted_full_relaxation` |

`full` and `full_direct` together give a direct comparison between relaxing from the raw inpainted geometry versus relaxing from an already-constrained geometry. All passes share the same `relax_config`; for the full passes, `elements_to_relax` is dropped automatically.

#### Post-relaxation steps

`refinement` and `uniqueness` run *after each active pass*, in order: relax → refine → deduplicate. Both are off by default — enable them with `include_task=True`.

```python
relaxation=InpaintingRelaxationConfig(
    # Which passes to run:
    constrained=True,      # relax only elements_to_relax
    full=True,             # then do a full relax on that output
    full_direct=False,     # skip direct full relax

    # Shared single-pass configuration:
    relax_config=RelaxationGraphConfig(
        params=RelaxationParams(
            mlip="mattersim",
            optimizer="BFGS",
            load_path="MatterSim-v1.0.0-1M",   # MLIP checkpoint
            fmax=0.05,
            max_n_steps=500,
            elements_to_relax=["H"],           # required when constrained=True
            return_initial_energies=False,
            return_final_forces=False,
        ),
        # Post-relaxation processing (applied to each pass):
        refinement=RefinementConfig(include_task=True, symprec=0.01),
        uniqueness=UniquenessConfig(
            include_task=True,
            symprec=0.01,
            ltol=0.2,
            stol=0.3,
            angle_tol=5.0,
        ),
        # AiiDA options (required field; pass None outside AiiDA):
        aiida=RelaxationAiiDAOptions(relax_code_label="xtalpaint@hpc"),
    ),
)
```

!!! warning "Constraints for `constrained`"
    `constrained=True` requires `relax_config.params.elements_to_relax` to be set.
    `full=True` requires `constrained=True` (the full-relax pass operates on the constrained output).
    At least one of the three pass flags must be `True`.

---

## Running without AiiDA

Without AiiDA, pass the `inpainting` config directly to the pipeline functions, together with the structures. The `aiida` block is simply not set (and `relax_config.aiida` is passed as `None`).

```python
from xtalpaint.inpainting.config_schema import XtalPaintConfig, InpaintingConfig
from xtalpaint.inpainting.inpainting_process import run_inpainting_pipeline
from xtalpaint.utils.relaxation_utils import relax_structures

config = XtalPaintConfig(
    candidate_generation=CandidateGenerationConfig(n_inp=2, element="H"),
    inpainting=InpaintingConfig(
        pretrained_name="TD-pos-only",   # auto-downloaded from Hugging Face
        predictor_corrector="TD",
        N_steps=5,
        coordinates_snr=0.2,
        n_corrector_steps=1,
        batch_size=1000,
        sampling_config_path="/path/to/sampling_conf",
    ),
)

# Run inpainting
results = run_inpainting_pipeline(
    structures={"host_001": structure},
    config=config.inpainting,          # pass InpaintingConfig directly
)
inpainted = results["structures"]

# Optional relaxation, reusing the relaxation params from the config
if config.relaxation is not None:
    relaxed_structures, final_energies, *_ = relax_structures(
        list(inpainted.values()),
        **config.relaxation.relax_config.params.model_dump(),
    )
```

!!! tip
    `InpaintingConfig.model_dump(exclude_none=True)` produces a plain dict that the pipeline functions also accept, which is convenient when serialising configs to JSON/YAML.

---

## Running with AiiDA

Add the `aiida` block to `XtalPaintConfig` for pipeline-level tasks (inpainting, candidate generation, pre-refinement). Relaxation AiiDA options live inside `relaxation.relax_config.aiida` — this keeps all relaxation settings in one place.

```python
from xtalpaint.inpainting.config_schema import (
    AiiDAOptions, AiiDATaskOptions, RelaxationAiiDAOptions,
)
from xtalpaint.aiida.workgraphs.inpainting import InpaintingWorkGraph

config = XtalPaintConfig(
    candidate_generation=...,
    inpainting=...,

    relaxation=InpaintingRelaxationConfig(
        constrained=True,
        full=True,
        relax_config=RelaxationGraphConfig(
            params=RelaxationParams(...),
            refinement=RefinementConfig(include_task=True),
            uniqueness=UniquenessConfig(include_task=True),
            # AiiDA options for relaxation tasks live here:
            aiida=RelaxationAiiDAOptions(
                relax_code_label="xtalpaint@hpc",
                relax_options=AiiDATaskOptions(
                    resources={"num_machines": 2, "num_mpiprocs_per_machine": 8},
                    withmpi=True,
                ),
            ),
        ),
    ),

    # Pipeline-level AiiDA options (inpainting, candidate gen, pre-refinement):
    aiida=AiiDAOptions(
        default_code_label="xtalpaint@localhost",
        inpainting_options=AiiDATaskOptions(
            resources={"num_machines": 1, "num_mpiprocs_per_machine": 4},
            max_wallclock_seconds=3600,
            withmpi=True,
        ),
    ),
)

# Build and submit the WorkGraph
wg = InpaintingWorkGraph.build(structures=structures, inputs=config)
wg.submit()
```

### Code label resolution

- **Pipeline tasks** (inpainting, candidate generation, pre-refinement): resolved from `XtalPaintConfig.aiida` — task-specific label, then `default_code_label` as fallback.
- **Relaxation tasks**: resolved from `relax_config.aiida` with *no* fallback between tasks — `relax_code_label` is required, and `refinement_code_label` / `uniqueness_code_label` must be set explicitly when the corresponding `include_task` flag is enabled.

### `AiiDATaskOptions` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `resources` | `dict` | `{}` | AiiDA scheduler resource dict |
| `max_wallclock_seconds` | `int` *(optional)* | — | Wall-clock limit |
| `queue_name` | `str` *(optional)* | — | Scheduler queue/partition |
| `withmpi` | `bool` | `False` | Enable MPI-parallel execution |

### `RelaxationAiiDAOptions` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `relax_code_label` | `str` | *required* | Code label for the relaxation task |
| `refinement_code_label` | `str \| None` | `None` | Code label for post-refinement |
| `uniqueness_code_label` | `str \| None` | `None` | Code label for the uniqueness filter |
| `relax_options` | `AiiDATaskOptions` | `{resources: {}, withmpi: False}` | Scheduler options for the relaxation task |
| `refinement_options` | `AiiDATaskOptions` | `{resources: {}, withmpi: False}` | Scheduler options for post-refinement |
| `uniqueness_options` | `AiiDATaskOptions` | `{resources: {}, withmpi: False}` | Scheduler options for the uniqueness filter |

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
        InpaintingRelaxationConfig,
        RelaxationGraphConfig,
        RelaxationParams,
        UniquenessConfig,
    )
    from xtalpaint.inpainting.inpainting_process import run_inpainting_pipeline

    structure = Structure.from_file("host.cif")

    config = XtalPaintConfig(
        candidate_generation=CandidateGenerationConfig(
            n_inp=2,
            element="H",
        ),
        inpainting=InpaintingConfig(
            pretrained_name="TD-pos-only",
            predictor_corrector="TD",
            N_steps=5,
            coordinates_snr=0.2,
            n_corrector_steps=1,
            batch_size=1000,
        ),
        pre_refinement=RefinementConfig(symprec=0.01),
        relaxation=InpaintingRelaxationConfig(
            constrained=True,
            relax_config=RelaxationGraphConfig(
                params=RelaxationParams(
                    mlip="mattersim",
                    optimizer="BFGS",
                    load_path="MatterSim-v1.0.0-1M",
                    elements_to_relax=["H"],
                    fmax=0.05,
                ),
                refinement=RefinementConfig(include_task=True, symprec=0.01),
                uniqueness=UniquenessConfig(include_task=True),
                aiida=None,   # plain Python execution
            ),
        ),
        # no aiida block → plain Python execution
    )

    results = run_inpainting_pipeline(
        structures={"host": structure},
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
        InpaintingRelaxationConfig,
        RelaxationGraphConfig,
        RelaxationParams,
        UniquenessConfig,
        RelaxationAiiDAOptions,
        AiiDAOptions,
        AiiDATaskOptions,
    )
    from xtalpaint.aiida.workgraphs.inpainting import InpaintingWorkGraph
    from xtalpaint.data import BatchedStructures

    structure = Structure.from_file("host.cif")

    config = XtalPaintConfig(
        candidate_generation=CandidateGenerationConfig(
            n_inp=2,
            element="H",
        ),
        inpainting=InpaintingConfig(
            pretrained_name="TD-pos-only",
            predictor_corrector="TD",
            N_steps=5,
            coordinates_snr=0.2,
            n_corrector_steps=1,
            batch_size=1000,
        ),
        pre_refinement=RefinementConfig(symprec=0.01),
        relaxation=InpaintingRelaxationConfig(
            constrained=True,
            relax_config=RelaxationGraphConfig(
                params=RelaxationParams(
                    mlip="mattersim",
                    optimizer="BFGS",
                    load_path="MatterSim-v1.0.0-1M",
                    elements_to_relax=["H"],
                    fmax=0.05,
                ),
                refinement=RefinementConfig(include_task=True, symprec=0.01),
                uniqueness=UniquenessConfig(include_task=True),
                aiida=RelaxationAiiDAOptions(
                    relax_code_label="xtalpaint@hpc",
                    refinement_code_label="xtalpaint@hpc",
                    uniqueness_code_label="xtalpaint@hpc",
                    relax_options=AiiDATaskOptions(
                        resources={"num_machines": 2, "num_mpiprocs_per_machine": 8},
                        withmpi=True,
                    ),
                ),
            ),
        ),
        aiida=AiiDAOptions(
            default_code_label="xtalpaint@localhost",
            inpainting_options=AiiDATaskOptions(
                resources={"num_machines": 1, "num_mpiprocs_per_machine": 4},
                withmpi=True,
            ),
        ),
    )

    wg = InpaintingWorkGraph.build(
        structures=BatchedStructures({"host": structure}),
        inputs=config,
    )
    wg.submit()
    ```

The two snippets are identical except for the `aiida=` blocks and the entry point. Develop and test workflows locally (without AiiDA) and then promote them to a remote HPC environment by adding the AiiDA blocks — no other changes needed.

---

## Configuration reference summary

| Class | Required fields | Purpose |
|---|---|---|
| `XtalPaintConfig` | `inpainting` | Top-level workflow config |
| `CandidateGenerationConfig` | `n_inp`, `element` | Generate inpainting masks |
| `InpaintingConfig` | `predictor_corrector`, `N_steps`, `coordinates_snr`, `n_corrector_steps`, `batch_size`, one of `pretrained_name`/`model_path` | Diffusion sampling |
| `RefinementConfig` | — | Symmetry refinement; toggled with `include_task` inside `RelaxationGraphConfig` |
| `RelaxationGraphConfig` | `params`, `aiida` (may be `None`) | Single-pass input for `relaxation_graph` |
| `InpaintingRelaxationConfig` | `relax_config` | Multi-pass relaxation stage (wraps a `RelaxationGraphConfig`) |
| `RelaxationParams` | `mlip`, `optimizer`, `load_path` | Inputs forwarded to `relax_structures()` |
| `UniquenessConfig` | — | Deduplication tolerances; toggled with `include_task` |
| `RelaxationAiiDAOptions` | `relax_code_label` | Code labels + scheduler options for relaxation tasks |
| `AiiDAOptions` | — | Code labels + scheduler options for pipeline tasks |
| `AiiDATaskOptions` | — | Resources, wall-clock, MPI flag |

### Using `relaxation_graph` directly

`relaxation_graph` accepts `RelaxationGraphConfig` directly, so you can call it outside the inpainting WorkGraph without needing the full `InpaintingRelaxationConfig`:

```python
from xtalpaint.aiida.workgraphs.relaxation import relaxation_graph
from xtalpaint.inpainting.config_schema import (
    RelaxationGraphConfig, RelaxationParams, RefinementConfig,
    UniquenessConfig, RelaxationAiiDAOptions, AiiDATaskOptions,
)

relax_cfg = RelaxationGraphConfig(
    params=RelaxationParams(
        mlip="mattersim",
        optimizer="BFGS",
        load_path="MatterSim-v1.0.0-1M",
        elements_to_relax=["H"],   # omit (or set to None) for a full relaxation
    ),
    refinement=RefinementConfig(include_task=True, symprec=0.01),
    uniqueness=UniquenessConfig(include_task=True),
    aiida=RelaxationAiiDAOptions(
        relax_code_label="xtalpaint@hpc",
        relax_options=AiiDATaskOptions(withmpi=True, resources={"num_machines": 1}),
    ),
)

wg = relaxation_graph.build(
    structures=my_structures,
    relax_config=relax_cfg,
)
wg.submit()
```

There is no separate `constrained` switch: the relaxation is constrained whenever `params.elements_to_relax` is set, and a full relaxation otherwise. To reuse the relaxation settings from an `InpaintingRelaxationConfig`, pass its `relax_config` attribute.
