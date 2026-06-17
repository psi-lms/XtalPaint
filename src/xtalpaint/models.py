"""Download and resolve the retrained XtalPaint model checkpoints.

The retrained MatterGen checkpoints used in our work are hosted on Hugging
Face (https://huggingface.co/t-reents/XtalPaint) rather than bundled with the
package. The ``TD-pos-only`` model is the recommended core model of XtalPaint;
``pos-only`` is also provided for comparison. Both are downloaded automatically
(and cached) when selected by name via ``pretrained_name``, mirroring how
MatterGen resolves its own pretrained checkpoints.
"""

from pathlib import Path
from typing import Optional

XTALPAINT_HF_REPO = "t-reents/XtalPaint"
XTALPAINT_PRETRAINED_MODELS = ("TD-pos-only", "pos-only")
RECOMMENDED_MODEL = "TD-pos-only"


def download_pretrained_model(
    model_name: str = RECOMMENDED_MODEL,
    local_dir: Optional[str] = None,
    repository_name: str = XTALPAINT_HF_REPO,
) -> Path:
    """Download a retrained XtalPaint checkpoint from Hugging Face.

    Args:
        model_name: Model to download, one of ``XTALPAINT_PRETRAINED_MODELS``.
        local_dir: Directory to download into. If ``None``, the shared Hugging
            Face cache is used.
        repository_name: Hugging Face repository to download from.

    Returns:
        Path to the local model directory containing ``config.yaml`` and
        ``checkpoints/last.ckpt``, usable as ``model_path``.
    """
    from huggingface_hub import hf_hub_download

    if model_name not in XTALPAINT_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown XtalPaint model '{model_name}'. Available models: "
            f"{list(XTALPAINT_PRETRAINED_MODELS)}."
        )

    download_kwargs = {"repo_id": repository_name}
    if local_dir is not None:
        download_kwargs["local_dir"] = local_dir

    hf_hub_download(
        filename=f"{model_name}/checkpoints/last.ckpt", **download_kwargs
    )
    config_path = hf_hub_download(
        filename=f"{model_name}/config.yaml", **download_kwargs
    )

    return Path(config_path).parent


def _has_checkpoint(model_path: str) -> bool:
    """Return whether ``model_path`` is a directory containing a checkpoint.

    Mirrors how ``MatterGenCheckpointInfo`` consumes ``model_path``: it expects
    a directory holding ``config.yaml`` and one or more ``*.ckpt`` files.
    """
    path = Path(model_path)
    return path.is_dir() and any(path.rglob("*.ckpt"))


def _auto_download_hint(model_name: str = RECOMMENDED_MODEL) -> str:
    """Return a hint describing how to obtain ``model_name``."""
    return (
        f"The '{model_name}' model is hosted on Hugging Face "
        f"(https://huggingface.co/{XTALPAINT_HF_REPO}). Select it by name to "
        "download it automatically:\n\n"
        f'    InpaintingConfig(pretrained_name="{model_name}", ...)\n\n'
        "or download it explicitly and pass the returned path as "
        "`model_path`:\n\n"
        "    from xtalpaint.models import download_pretrained_model\n"
        f'    model_path = download_pretrained_model("{model_name}")'
    )


def resolve_inpainting_model(
    predictor_corrector: str,
    pretrained_name: Optional[str],
    model_path: Optional[str],
    local_dir: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the inpainting model, auto-downloading XtalPaint checkpoints.

    An XtalPaint model selected via ``pretrained_name`` is downloaded from
    Hugging Face and returned as a local ``model_path``. Otherwise the model
    selection is validated and passed through unchanged.

    Args:
        predictor_corrector: The selected predictor-corrector key.
        pretrained_name: A bundled MatterGen checkpoint or XtalPaint model
            name.
        model_path: Path to a local checkpoint directory.
        local_dir: Directory to download XtalPaint models into.

    Returns:
        A ``(pretrained_name, model_path)`` tuple ready for the generator.
        For an XtalPaint model, ``pretrained_name`` is ``None`` and
        ``model_path`` points to the downloaded checkpoint.

    Raises:
        FileNotFoundError: If ``model_path`` is set but the checkpoint is
            missing.
        ValueError: If the ``TD`` predictor-corrector is used without the
            ``TD-pos-only`` model.
    """
    if pretrained_name in XTALPAINT_PRETRAINED_MODELS:
        downloaded = download_pretrained_model(
            pretrained_name, local_dir=local_dir
        )
        return None, str(downloaded)

    if model_path is not None:
        if _has_checkpoint(model_path):
            return pretrained_name, model_path
        raise FileNotFoundError(
            f"No model checkpoint was found at '{model_path}'.\n\n"
            "This is expected if you have not downloaded the retrained "
            "XtalPaint model yet.\n\n" + _auto_download_hint(RECOMMENDED_MODEL)
        )

    if predictor_corrector == "TD":
        raise ValueError(
            "The 'TD' predictor-corrector requires the retrained "
            "'TD-pos-only' model, which is not bundled with MatterGen.\n\n"
            + _auto_download_hint("TD-pos-only")
        )

    return pretrained_name, model_path
