"""Patient manifest, explicit normalization, and paired/unpaired 2.5D loading."""

from collections import OrderedDict
from collections.abc import Mapping
import csv
import math
from numbers import Real
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import resize


SAMPLER_VERSION = "region_draw"


def sampler_version(training):
    grouped = training.get("group_batches_by_region", False)
    if type(grouped) is not bool:
        raise ValueError("group_batches_by_region must be a boolean")
    return "region_grouped_draw" if grouped else SAMPLER_VERSION


def read_manifest(path, split=None):
    path = Path(path).resolve()
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "patient_id", "region", "split", "paired", "mr", "ct"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires columns: {sorted(required)}")
    if split is not None and split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    seen, patient_splits, volume_splits = set(), {}, {}
    for row in rows:
        for key in ("case_id", "patient_id"):
            if not row.get(key) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row[key]):
                raise ValueError(f"{key} must be a nonempty safe identifier")
        if row["case_id"] in seen:
            raise ValueError("case_id must be unique")
        seen.add(row["case_id"])
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if (row.get("paired") or "").lower() not in {"true", "false", "0", "1"}:
            raise ValueError("paired must be true/false or 1/0")
        row["paired"] = row["paired"].lower() in {"true", "1"}
        ct_patient_id = row.get("ct_patient_id") or ""
        if ct_patient_id:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ct_patient_id):
                raise ValueError("ct_patient_id must be a safe identifier")
            if row["paired"] and ct_patient_id != row["patient_id"]:
                raise ValueError("Paired CT and MRI must have the same patient identity")
            patient_splits.setdefault(ct_patient_id, set()).add(row["split"])
        if not row.get("mr"):
            raise ValueError(f"MRI path is required: {row['case_id']}")
        for key in ("mr", "ct"):
            if row[key]:
                p = Path(row[key]).expanduser()
                row[key] = str((p if p.is_absolute() else path.parent / p).resolve())
                volume_splits.setdefault(row[key], set()).add(row["split"])
    if any(len(splits) > 1 for splits in patient_splits.values()):
        raise ValueError("A patient occurs in multiple splits")
    if any(len(splits) > 1 for splits in volume_splits.values()):
        raise ValueError("The same volume path occurs in multiple splits")
    selected = [row for row in rows if split is None or row["split"] == split]
    if not selected:
        raise ValueError(f"No {split} cases")
    return selected


def load_image(path, preprocessing):
    image = nib.load(path)
    if preprocessing.get("orientation", "native") == "RAS":
        image = nib.as_closest_canonical(image)
    elif preprocessing.get("orientation", "native") != "native":
        raise ValueError("orientation must be RAS or native")
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3 or not all(data.shape) or not np.isfinite(data).all():
        raise ValueError(f"Expected a finite 3-D NIfTI volume: {path}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError(f"Invalid physical image grid: {path}")
    return image, data


def normalize_mr(data, settings):
    mode = settings.get("mr_normalization", "percentile")
    if mode not in {"minmax", "percentile"}:
        raise ValueError("mr_normalization must be minmax or percentile")
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite, nonempty MRI data")
    if mode == "minmax":
        low, high = float(data.min()), float(data.max())
    else:
        values = (
            data[data != 0]
            if settings.get("mr_foreground_only", True)
            else data.ravel()
        )
        if not values.size:
            raise ValueError("Empty MRI foreground")
        percentiles = np.asarray(settings.get("mr_percentiles", []), dtype=float)
        if percentiles.shape != (2,) or not 0 <= percentiles[0] < percentiles[1] <= 100:
            raise ValueError("mr_percentiles must be two increasing values in [0, 100]")
        low, high = np.percentile(values, percentiles)
    if high <= low:
        raise ValueError("Degenerate MRI normalization range")
    return (np.clip((data - low) / (high - low), 0, 1) * 2 - 1).astype(np.float32)


def normalize_ct(data, settings):
    low, high = settings["ct_clip"]
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid CT clipping interval")
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite, nonempty CT data")
    return ((np.clip(data, low, high) - low) / (high - low) * 2 - 1).astype(np.float32)


def to_hu(data, settings):
    low, high = settings["ct_clip"]
    return (data + 1) * 0.5 * (high - low) + low


def validate_unpaired_ct_neighbor_strides(training, region_order, rows=()):
    """Validate CT neighbor-stride settings."""
    name = "unpaired_ct_neighbor_stride_by_region"
    strides = training.get(name, {})
    if not isinstance(strides, Mapping) or not set(strides).issubset(region_order):
        raise ValueError(f"{name} requires known region keys")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer)) or value < 1
           for value in strides.values()):
        raise ValueError(f"{name} values must be positive integers")
    if any(row["paired"] and strides.get(row["region"], 1) != 1 for row in rows):
        raise ValueError(f"{name} must remain 1 for every region containing paired cases")
    return {region: int(value) for region, value in strides.items()}


def slice_stack(data, center, channels, size, settings, *, neighbor_stride=1):
    if not isinstance(channels, int) or channels < 1 or channels % 2 != 1:
        raise ValueError("2.5D input requires a positive odd number of channels")
    if data.ndim != 3 or not all(data.shape):
        raise ValueError("Expected a nonempty 3-D volume")
    if not isinstance(size, int) or size < 1:
        raise ValueError("image size must be positive")
    if not isinstance(center, (int, np.integer)) or not 0 <= center < data.shape[2]:
        raise ValueError("Slice center is outside the volume")
    if (isinstance(neighbor_stride, (bool, np.bool_))
            or not isinstance(neighbor_stride, (int, np.integer)) or neighbor_stride < 1):
        raise ValueError("neighbor_stride must be a positive integer")
    backend = settings.get("resize_backend", "torchvision")
    if backend not in {"torchvision", "torchio"}:
        raise ValueError("resize_backend must be torchvision or torchio")
    if backend == "torchio" and settings.get("quantize_uint8", False):
        raise ValueError("TorchIO resizing requires floating-point input")
    if neighbor_stride == 1:
        # Select adjacent slices and clamp indices at the volume boundary.
        indices = np.clip(
            np.arange(center - channels // 2, center + channels // 2 + 1),
            0,
            data.shape[2] - 1,
        )
    else:
        # Select native slices at the requested stride.
        indices = [min(max(int(center) + offset * int(neighbor_stride), 0), data.shape[2] - 1)
                   for offset in range(-(channels // 2), channels // 2 + 1)]
    slices = []
    for z in indices:
        value = data[:, :, z]
        if settings.get("transpose_xy", True):
            value = value.T
        if settings.get("quantize_uint8", False):
            # Optional image quantization; the default uses float32.
            image = Image.fromarray((value + 1) * 0.5 * 255).convert("L")
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            value = np.asarray(image, dtype=np.float32) / 255 * 2 - 1
        slices.append(np.ascontiguousarray(value))
    tensor = torch.from_numpy(np.stack(slices))
    if backend == "torchio":
        try:
            import torchio as tio
        except ImportError as exc:
            raise ImportError(
                "TorchIO preprocessing requires: pip install -e '.[torchio]'"
            ) from exc
        # Use the [1, X, Y, slices] layout before converting to CHW.
        volume = tensor.permute(1, 2, 0).unsqueeze(0).numpy()
        resized = tio.Resize((size, size, channels))(volume)
        return (
            torch.as_tensor(resized[0], dtype=torch.float32)
            .permute(2, 0, 1)
            .contiguous()
        )
    if tensor.shape[-2:] != (size, size):
        tensor = resize(tensor, [size, size], antialias=True)
    return tensor


class SliceDataset(Dataset):
    def __init__(self, manifest, config, split="train"):
        from rsam.anatomy_losses import validate_body_envelope_config
        self.body_envelope_options = validate_body_envelope_config(config["training"])
        self.rows = read_manifest(manifest, split)
        if not self.rows:
            raise ValueError(f"No {split} cases")
        self.config = config
        self.split = split
        self.region_order = config["region_order"]
        self.preprocessing = config["preprocessing"]
        self.samples_per_case = config["training"]["slices_per_case"]
        self.epoch = 0
        self.cache = OrderedDict()
        self.mr_spacing = {}
        physical_margins = config["training"].get("exclude_edge_mm_by_region", {})
        if not isinstance(physical_margins, dict) or not set(physical_margins).issubset(self.region_order):
            raise ValueError("exclude_edge_mm_by_region requires known region keys")
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, float, np.integer, np.floating))
               or not np.isfinite(value) or value < 0 for value in physical_margins.values()):
            raise ValueError("exclude_edge_mm_by_region values must be finite and nonnegative")
        self.physical_margins = physical_margins
        self.unpaired_ct_neighbor_strides = validate_unpaired_ct_neighbor_strides(
            config["training"], self.region_order, self.rows
        )
        self.ct_pools = {}
        for row in self.rows:
            if row["region"] not in self.region_order or not row["mr"] or not row["ct"]:
                raise ValueError(
                    "Training requires a known region, MRI, and CT-domain reference"
                )
            pool = self.ct_pools.setdefault(row["region"], [])
            if row["ct"] not in pool:
                pool.append(row["ct"])
            if row["paired"]:
                a, b = nib.load(row["mr"]), nib.load(row["ct"])
                if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4, rtol=0):
                    raise ValueError(
                        f"Paired MRI and CT must be registered on one grid: {row['case_id']}"
                    )

    def __len__(self):
        return len(self.rows) * self.samples_per_case

    def _volume(self, path, kind):
        key = (path, kind)
        if key not in self.cache:
            image, data = load_image(path, self.preprocessing)
            if kind == "mr":
                self.mr_spacing[path] = float(image.header.get_zooms()[2])
            normalized = (
                normalize_mr(data, self.preprocessing)
                if kind == "mr"
                else normalize_ct(data, self.preprocessing)
            )
            self.cache[key] = normalized
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, index):
        draw_position = None
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("A sampled index must be (index, global_draw_position)")
            index, draw_position = index
            if (not isinstance(draw_position, (int, np.integer))
                    or isinstance(draw_position, (bool, np.bool_)) or draw_position < 0):
                raise ValueError("global_draw_position must be a nonnegative integer")
        if (not isinstance(index, (int, np.integer))
                or isinstance(index, (bool, np.bool_)) or not 0 <= index < len(self)):
            raise IndexError("Slice sample index must be an integer within the dataset")
        index = int(index)
        row = self.rows[index // self.samples_per_case]
        seed = self.config["training"]["seed"]
        if draw_position is None:
            # Seed direct samples by index.
            rng = np.random.default_rng(seed + self.epoch * 1000003 + index)
        else:
            # Seed repeated samples by draw position.
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, self.epoch, index, int(draw_position)]
            ))
        mr = self._volume(row["mr"], "mr")
        if row["paired"]:
            ct = self._volume(row["ct"], "ct")
        else:
            pool = self.ct_pools[row["region"]]
            ct = self._volume(pool[int(rng.integers(len(pool)))], "ct")
        margin = self.config["training"].get("exclude_edge_slices", 0)
        if not isinstance(margin, int) or margin < 0:
            raise ValueError("exclude_edge_slices must be a nonnegative integer")
        if row["region"] in self.physical_margins:
            spacing = self.mr_spacing[row["mr"]]
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError("MRI slice spacing must be finite and positive")
            margin = math.ceil(self.physical_margins[row["region"]] / spacing)
        margin = min(margin, (mr.shape[2] - 1) // 2)
        lo, hi = margin, mr.shape[2] - margin
        center = int(rng.integers(lo, hi))
        center_ct = center if row["paired"] else int(rng.integers(ct.shape[2]))
        model = self.config["model"]
        a = slice_stack(
            mr, center, model["input_nc"], model["image_size"], self.preprocessing
        )
        ct_stride = (self.unpaired_ct_neighbor_strides.get(row["region"], 1)
                     if self.split == "train" and not row["paired"] else 1)
        if ct_stride == 1:
            b = slice_stack(
                ct, center_ct, model["output_nc"], model["image_size"], self.preprocessing
            )
        else:
            b = slice_stack(
                ct, center_ct, model["output_nc"], model["image_size"], self.preprocessing,
                neighbor_stride=ct_stride,
            )
        if self.config["training"].get("horizontal_flip", True) and rng.random() < 0.5:
            a, b = a.flip(-1), b.flip(-1)
        code = torch.zeros(len(self.region_order))
        code[self.region_order.index(row["region"])] = 1
        sample = {
            "mr": a,
            "ct": b,
            "code": code,
            "paired": row["paired"],
            "case_id": row["case_id"],
        }
        training = self.config["training"]
        if (training.get("paired_reconstruction_mode", "full_fov") != "full_fov"
                or training.get("unpaired_structure_weight", 0)
                or training.get("unpaired_air_weight", 0)
                or training.get("paired_gradient_weight", 0)
                or training.get("paired_mse_weight", 0)
                or training.get("ct_critic_crop_regions")):
            from rsam.anatomy_losses import body_envelope
            sample["mr_body"] = body_envelope(a, -0.95, **self.body_envelope_options)
            low, high = self.preprocessing["ct_clip"]
            sample["ct_body"] = (body_envelope(b, 2 * (-500 - low) / (high - low) - 1,
                                               **self.body_envelope_options)
                                 if row["paired"] or row["region"] in training.get("ct_critic_crop_regions", [])
                                 else torch.ones_like(b, dtype=torch.bool))
        return sample


class RegionSampler(Sampler):
    """Sample seeded region-balanced draws and shard them across ranks."""

    version = SAMPLER_VERSION

    def __init__(self, dataset, rank=0, world_size=1, balanced=True):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be an integer in [0, world_size)")
        if type(balanced) is not bool:
            raise ValueError("balanced must be a boolean")
        if len(dataset) < 1:
            raise ValueError("Cannot sample an empty dataset")
        seed = dataset.config["training"]["seed"]
        if (not isinstance(seed, (int, np.integer))
                or isinstance(seed, (bool, np.bool_)) or seed < 0):
            raise ValueError("Sampling seed must be a nonnegative integer")
        self.dataset, self.rank, self.world_size = dataset, rank, world_size
        self.balanced, self.epoch = balanced, 0
        training = dataset.config["training"]
        self.version = sampler_version(training)
        self.group_batches = training.get("group_batches_by_region", False)
        self.global_batch_size = training.get("global_batch_size")
        if self.group_batches and (
                type(self.global_batch_size) is not int or self.global_batch_size <= 0
                or self.global_batch_size % world_size):
            raise ValueError("Grouped sampling requires a positive global_batch_size divisible by world_size")
        self.num_samples = math.ceil(len(dataset) / world_size)
        counts = {
            region: sum(row["region"] == region for row in dataset.rows)
            for region in dataset.region_order
        }
        counts = {region: count for region, count in counts.items() if count > 0}
        if sum(counts.values()) != len(dataset.rows):
            raise ValueError("Sample rows contain an unknown region")
        requested = dataset.config["training"].get("region_sampling_weights")
        if requested is not None:
            if not balanced:
                raise ValueError("region_sampling_weights requires region_balanced_sampling")
            if not isinstance(requested, Mapping) or set(requested) != set(counts):
                raise ValueError("region_sampling_weights must name exactly the active regions")
            if any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, Real)
                   or not math.isfinite(value) or value <= 0
                   for value in requested.values()):
                raise ValueError("region_sampling_weights must be positive finite numbers")
        region_weights = requested or {region: 1.0 for region in counts}
        mass = region_weights if balanced else counts
        try:
            total_mass = math.fsum(mass.values())
        except OverflowError as exc:
            raise ValueError("The sum of region_sampling_weights must be finite") from exc
        if not math.isfinite(total_mass):
            raise ValueError("The sum of region_sampling_weights must be finite")
        self.region_probabilities = {
            region: float(value / total_mass) for region, value in mass.items()
        }
        # Scale relative weights to prevent overflow.
        weight_scale = max(region_weights.values())
        self._sample_weights = torch.tensor([
            (region_weights[row["region"]] / weight_scale) / counts[row["region"]]
            for row in dataset.rows
            for _ in range(dataset.samples_per_case)
        ], dtype=torch.double)
        if not torch.isfinite(self._sample_weights).all() or not (self._sample_weights > 0).all():
            raise ValueError("Regional sample weights must be representable positive finite numbers")

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch
        self.dataset.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.dataset.config["training"]["seed"] + self.epoch
        )
        total = self.num_samples * self.world_size
        if self.balanced:
            indices = torch.multinomial(
                self._sample_weights,
                total,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
            indices = (indices * math.ceil(total / len(indices)))[:total]
        positions = list(range(total))
        if self.group_batches:
            # Preserve draw tokens while grouping samples by region.
            pools = {region: [] for region in self.dataset.region_order}
            for position, index in enumerate(indices):
                region = self.dataset.rows[index // self.dataset.samples_per_case]["region"]
                pools[region].append(position)
            batches, tails = [], []
            size = self.global_batch_size
            for pool in pools.values():
                full = len(pool) // size * size
                batches.extend(pool[start:start + size] for start in range(0, full, size))
                tails.extend(pool[full:])
            full = len(tails) // size * size
            batches.extend(tails[start:start + size] for start in range(0, full, size))
            order = torch.randperm(len(batches), generator=generator).tolist()
            positions = [position for index in order for position in batches[index]]
            # Append the incomplete batch after the full batches.
            positions.extend(tails[full:])
        return (
            (indices[positions[offset]], positions[offset])
            for offset in range(self.rank, total, self.world_size)
        )
