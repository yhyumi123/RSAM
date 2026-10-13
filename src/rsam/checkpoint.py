"""Save and load checkpoints with model and preprocessing metadata."""

import hashlib
import os
from pathlib import Path
import tempfile

import torch

from rsam.model import Generator, ModelConfig


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def atomic_save(value, path):
    """Write a checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_generator(path, model, preprocessing, region_order, provenance=None):
    model = unwrap(model)
    if (len(region_order) != model.config.num_regions
            or len(set(region_order)) != len(region_order)):
        raise ValueError("region_order must contain one unique name per model region")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(
        {
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "preprocessing": preprocessing,
            "region_order": region_order,
            "provenance": provenance or {},
        },
        path,
    )


def load_generator(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not {"model_config", "state_dict", "preprocessing", "region_order"}.issubset(
        bundle
    ):
        raise ValueError(
            "Expected a generator checkpoint with architecture and preprocessing metadata"
        )
    if bundle.get("format_version", 1) != 1:
        raise ValueError("Unsupported generator checkpoint format_version")
    configuration = dict(bundle["model_config"])
    # Set defaults for optional model metadata.
    configuration.setdefault("attention_backend", "eager")
    configuration.setdefault("rsam_second_relu", False)
    configuration.setdefault("stable_instance_norm", False)
    model = Generator(ModelConfig(**configuration))
    if (len(bundle["region_order"]) != model.config.num_regions
            or len(set(bundle["region_order"])) != model.config.num_regions):
        raise ValueError("Checkpoint region_order does not match its model")
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(device).eval()
    return model, bundle

