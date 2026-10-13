"""MR-to-CT NIfTI inference, retaining the loaded physical image grid."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms.functional import resize

from rsam.checkpoint import load_generator, sha256
from rsam.data import load_image, normalize_mr, read_manifest, slice_stack, to_hu


@torch.inference_mode()
def predict_volume(model, bundle, mr_path, region, device, batch_size=1):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    settings = bundle["preprocessing"]
    native_image = nib.load(mr_path)
    image, raw = load_image(mr_path, settings)
    normalized = normalize_mr(raw, settings)
    result = np.empty_like(raw, dtype=np.float32)
    region_index = bundle["region_order"].index(region)
    for start in range(0, raw.shape[2], batch_size):
        indices = list(range(start, min(raw.shape[2], start + batch_size)))
        stacks = [
            slice_stack(
                normalized, z, model.config.input_nc, model.config.image_size, settings
            )
            for z in indices
        ]
        tensor = torch.stack(stacks).to(device)
        codes = torch.eye(model.config.num_regions, device=device)[
            region_index : region_index + 1
        ].expand(len(indices), -1)
        predicted = model(tensor, codes)[:, model.config.output_nc // 2]
        shape = (
            (raw.shape[1], raw.shape[0])
            if settings.get("transpose_xy", True)
            else raw.shape[:2]
        )
        predicted = (
            resize(predicted.unsqueeze(1), list(shape), antialias=True)[:, 0]
            .cpu()
            .numpy()
        )
        for offset, z in enumerate(indices):
            values = (
                predicted[offset].T
                if settings.get("transpose_xy", True)
                else predicted[offset]
            )
            result[:, :, z] = to_hu(values, settings)
    if settings.get("orientation", "native") == "RAS":
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine),
            nib.orientations.io_orientation(native_image.affine),
        )
        result = nib.orientations.apply_orientation(result, transform)
    if result.shape != native_image.shape:
        raise ValueError("Prediction could not be restored to the native MRI grid")
    header = native_image.header.copy()
    header.set_data_dtype(np.float32)
    # Preserve the native image grid in the output header.
    return nib.Nifti1Image(result, native_image.affine, header)


def infer_manifest(
    checkpoint,
    manifest,
    output,
    split="test",
    device="cpu",
    batch_size=1,
    rank=0,
    world_size=1,
):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    model, bundle = load_generator(checkpoint, device)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest, split)[rank::world_size]
    checkpoint_hash = sha256(checkpoint)
    records = []
    for row in rows:
        image = predict_volume(
            model, bundle, row["mr"], row["region"], device, batch_size
        )
        image.set_data_dtype(np.float32)
        path = root / f"{row['case_id']}_sct.nii.gz"
        nib.save(image, path)
        records.append(
            {
                "case_id": row["case_id"],
                "region": row["region"],
                "output": path.name,
                "checkpoint_sha256": checkpoint_hash,
                "source_mr_sha256": sha256(row["mr"]),
                "output_sha256": sha256(path),
                "shape": list(image.shape),
                "affine": image.affine.tolist(),
            }
        )
        print(
            json.dumps({"rank": rank, "case_id": row["case_id"], "output": path.name}),
            flush=True,
        )
    (root / f"inference_rank{rank}.json").write_text(json.dumps(records, indent=2))
