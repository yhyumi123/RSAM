"""Single-GPU and distributed training with epoch-boundary resume."""

from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import yaml

from rsam.checkpoint import atomic_save, save_generator, sha256, unwrap
from rsam.data import (
    RegionSampler, SliceDataset, sampler_version, validate_unpaired_ct_neighbor_strides,
)
from rsam.losses import (
    ReplayBuffer, VGGPerceptualLoss, adversarial_loss, distributed_replay_query,
    validate_vgg_target_mode, paired_reconstruction_loss,
)
from rsam.contrastive import ProjectionHeads, patch_nce
from rsam.critic import ct_critic_input
from rsam.model import Discriminator, Generator, ModelConfig, initialize_weights
from rsam.anatomy_losses import (body_reconstruction_loss, unpaired_anatomy_losses,
    paired_gradient_loss, paired_foreground_mse_loss, validate_body_envelope_config)


class CycleGenerators(nn.Module):
    def __init__(self, config, nce_enabled=False):
        super().__init__()
        self.mr_to_ct = Generator(ModelConfig(**config))
        reverse = {
            **config,
            "input_nc": config["output_nc"],
            "output_nc": config["input_nc"],
        }
        self.ct_to_mr = Generator(ModelConfig(**reverse))
        self.projection_heads = ProjectionHeads(config["base_channels"]) if nce_enabled else None

    def forward(self, mr, ct, code, step=0, seed=0):
        fake_ct, fake_mr = self.mr_to_ct(mr, code), self.ct_to_mr(ct, code)
        generated = {
            "fake_ct": fake_ct,
            "fake_mr": fake_mr,
            "cycle_mr": self.ct_to_mr(fake_ct, code),
            "cycle_ct": self.mr_to_ct(fake_mr, code),
            "identity_mr": self.ct_to_mr(mr, code),
            "identity_ct": self.mr_to_ct(ct, code),
        }

        if self.projection_heads is not None:
            generated["nce_loss"] = 0.5 * (
                patch_nce(self.mr_to_ct, self.projection_heads, mr, fake_ct, step, seed, 0)
                + patch_nce(self.mr_to_ct, self.projection_heads, ct, generated["identity_ct"], step, seed, 1)
            )
        return generated


class Discriminators(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        stable = model_config.get("stable_instance_norm", False)
        self.mr = Discriminator(
            model_config["input_nc"], model_config["base_channels"], stable
        )
        self.ct = Discriminator(
            model_config["output_nc"], model_config["base_channels"], stable
        )

    def forward(self, mr, ct, fake_mr=None, fake_ct=None):
        if fake_mr is None:
            return self.mr(mr), self.ct(ct)
        a = self.mr(torch.cat([mr, fake_mr], dim=0)).chunk(2)
        b = self.ct(torch.cat([ct, fake_ct], dim=0)).chunk(2)
        return a[0], a[1], b[0], b[1]


def resolve_batch_size(config, world_size):
    """Compute the local batch size from the global batch and process count."""
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    per_gpu, global_batch = config.get("batch_size_per_gpu"), config.get("global_batch_size")
    for name, value in (("batch_size_per_gpu", per_gpu), ("global_batch_size", global_batch)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if global_batch is not None:
        if global_batch % world_size:
            raise ValueError("global_batch_size must be divisible by WORLD_SIZE")
        derived = global_batch // world_size
        if per_gpu is not None and per_gpu != derived:
            raise ValueError("global_batch_size disagrees with batch_size_per_gpu * WORLD_SIZE")
        return derived
    if per_gpu is None:
        raise ValueError("Specify global_batch_size or batch_size_per_gpu")
    return per_gpu


def validate_training_config(config):
    model = ModelConfig(**config["model"])
    regions = config["region_order"]
    if len(regions) != model.num_regions or len(set(regions)) != len(regions):
        raise ValueError("region_order must contain one unique name per model region")
    if model.input_nc != model.output_nc:
        raise ValueError("Cycle identity loss requires equal modality channel counts")
    cfg = config["training"]
    validate_body_envelope_config(cfg)
    validate_unpaired_ct_neighbor_strides(cfg, regions)
    if model.stable_instance_norm and cfg.get("precision", "fp32") != "fp32":
        raise ValueError("stable_instance_norm requires training.precision=fp32")
    sampler_version(cfg)
    crop_regions = cfg.get("ct_critic_crop_regions", [])
    if not isinstance(crop_regions, list) or any(r not in regions for r in crop_regions):
        raise ValueError("ct_critic_crop_regions must list known regions")
    if crop_regions and cfg.get("replay_size", 50) != 0:
        raise ValueError("CT critic cropping requires replay_size=0 to retain matching body masks")
    ratio = cfg.get("projection_lr_multiplier", 1.)
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("projection_lr_multiplier must be finite and positive")
    if cfg.get("nce_weight", 0) and model.n_residual_blocks < 1:
        raise ValueError("PatchNCE requires at least one residual block")
    if cfg.get("warm_start_training_state"):
        raise ValueError("Use --resume with a full training checkpoint, not an implicit warm-start path")
    if (cfg.get("perceptual_target_policy", "all_reference") != "all_reference"
            or cfg.get("unpaired_perceptual_scale", 1) != 1):
        raise ValueError("Unsupported perceptual policy: use explicit validated objective coefficients")
    if cfg.get("vgg_slice_mode", "central") == "stack" and model.output_nc != 3:
        raise ValueError("stack VGG supervision requires three output slices")
    for name in ("unpaired_structure_weight", "unpaired_air_weight", "paired_gradient_weight", "paired_mse_weight",
                 "paired_full_fov_weight", "paired_background_weight", "nce_weight"):
        value = cfg.get(name, 0.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if cfg.get("paired_reconstruction_mode", "full_fov") not in {"full_fov", "body_plus_background"}:
        raise ValueError("Unknown paired reconstruction mode")
    anchor_weight = cfg.get("paired_reconstruction_weight", 0.)
    if not isinstance(anchor_weight, (int, float)) or not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError("paired_reconstruction_weight must be finite and nonnegative")
    if cfg.get("discriminator_aggregation", "patch") not in {"patch", "global_mean"}:
        raise ValueError("discriminator_aggregation must be patch or global_mean")
    validate_vgg_target_mode(cfg.get("vgg_target_mode", "spatial"),
                             cfg.get("vgg_criterion", "l1"),
                             cfg.get("vgg_feature_reduction", "mean"))
    if type(cfg.get("require_new_run_initialization", False)) is not bool:
        raise ValueError("require_new_run_initialization must be a boolean")
    if cfg.get("replay_scope", "per_rank") not in {"per_rank", "global"}:
        raise ValueError("replay_scope must be per_rank or global")
    interval = cfg.get("save_every_epochs", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("save_every_epochs must be a nonnegative integer")
    if type(cfg.get("deterministic", False)) is not bool:
        raise ValueError("deterministic must be a boolean")
    for name in ("epochs", "slices_per_case", "cpu_threads"):
        value = cfg.get(name, 2 if name == "cpu_threads" else None)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("num_workers", "decay_start_epoch"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if cfg["decay_start_epoch"] >= cfg["epochs"]:
        raise ValueError("decay_start_epoch must precede epochs")
    if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
        raise ValueError("lr must be finite and positive")
    if len(cfg["betas"]) != 2 or any(not 0 <= beta < 1 for beta in cfg["betas"]):
        raise ValueError("Adam betas must contain two values in [0, 1)")
    for name in ("adversarial", "cycle", "identity", "perceptual"):
        value = cfg["loss_weights"][name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Loss weight {name} must be finite and nonnegative")
    if cfg["gan_loss"] not in {"logistic", "lsgan"}:
        raise ValueError("gan_loss must be logistic or lsgan")


def configure_determinism(enabled):
    """Configure deterministic operations."""
    if enabled:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization "
                    "when deterministic training is requested"
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    # Enforce deterministic operations.
    torch.use_deterministic_algorithms(enabled, warn_only=False)


def save_epoch_snapshot(output, generator, config, provenance, completed_epochs):
    """Save an MR-to-CT generator at an epoch boundary."""
    path = Path(output) / "snapshots" / f"generator_mr_to_ct_epoch{completed_epochs:04d}.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing epoch snapshot: {path}")
    save_generator(path, generator, config["preprocessing"], config["region_order"], {
        **provenance,
        "epochs_completed": completed_epochs,
        "snapshot_kind": "initialization" if completed_epochs == 0 else "epoch_boundary",
    })
    return path


def train(config_path, manifest, output, resume=None, max_steps=None, device_name=None):
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_training_config(config)
    cfg = config["training"]
    active_sampler_version = sampler_version(cfg)
    replay_scope = cfg.get("replay_scope", "per_rank")
    save_interval = cfg.get("save_every_epochs", 0)
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps must be a positive total step limit")
    world = int(os.environ.get("WORLD_SIZE", 1))
    batch_size = resolve_batch_size(cfg, world)
    rank, local_rank = (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )
    output = Path(output).resolve()
    if (resume and cfg.get("require_new_run_initialization", False)
            and Path(resume).resolve().parent != output):
        raise ValueError("Resume must use a checkpoint created inside this new run")
    if output.exists() and (not output.is_dir() or (not resume and any(output.iterdir()))):
        raise ValueError("Output is occupied; choose an empty directory or use --resume")
    if (resume and output.exists() and any(output.iterdir())
            and Path(resume).resolve().parent != output):
        raise ValueError("An occupied output must resume its own checkpoint directory")
    manifest_hash = sha256(manifest)
    state, epoch_start, steps = None, 0, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("sampler_version") != active_sampler_version:
            raise ValueError(
                "Resume requires the same sampler_version as the current training configuration"
            )
        if (state["config"] != config or state["manifest_sha256"] != manifest_hash
                or state["world_size"] != world):
            raise ValueError("Resume requires the same config, manifest, and world size")
        if state.get("replay_scope", "per_rank") != replay_scope:
            raise ValueError("Resume requires the same replay_scope")
        epoch_start, steps = state["next_epoch"], state["steps"]
        if epoch_start >= cfg["epochs"]:
            raise ValueError("The resumed run already completed its configured epochs")
        if max_steps is not None and steps >= max_steps:
            raise ValueError("The resumed run already reached max_steps")
    deterministic = cfg.get("deterministic", False)
    configure_determinism(deterministic)
    device = torch.device(
        device_name or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", local_rank)
        if world > 1 and device.index != local_rank:
            raise ValueError("For torchrun, the CUDA device must match LOCAL_RANK")
        torch.cuda.set_device(device)
    if world > 1:
        process_options = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(minutes=3),
            **process_options,
        )
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    torch.manual_seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = SliceDataset(manifest, config)
    ct_context_provenance = {
        "unpaired_ct_neighbor_stride_by_region": {
            region: dataset.unpaired_ct_neighbor_strides.get(region, 1)
            for region in config["region_order"]
        },
        "unpaired_ct_neighbor_boundary": "clamp",
        "unpaired_ct_neighbor_scope": "training_unpaired_CT_only",
        "unpaired_ct_center_sampling": "unchanged_uniform_native_depth",
    }
    sampler = RegionSampler(
        dataset, rank, world, cfg.get("region_balanced_sampling", True)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    effective_model_config = dict(config["model"])
    if state is not None:
        effective_model_config.setdefault(
            "rsam_second_relu", state.get("model_config", {}).get("rsam_second_relu", False)
        )
    if state is not None and "rank_state_files" not in state:
        effective_model_config.setdefault("attention_backend", "eager")
    nce_weight = cfg.get("nce_weight", 0.)
    full_fov_weight = cfg.get("paired_full_fov_weight", 0.)
    crop_indices = [config["region_order"].index(r) for r in cfg.get("ct_critic_crop_regions", [])]
    generators = CycleGenerators(effective_model_config, nce_enabled=bool(nce_weight)).to(device)
    discriminators = Discriminators(config["model"]).to(device)
    for network in (generators, discriminators):
        network.apply(
            lambda layer: initialize_weights(
                layer, cfg.get("initialization", "kaiming")
            )
        )
    perceptual_weight = cfg["loss_weights"]["perceptual"]
    anchor_weight = cfg.get("paired_reconstruction_weight", 0.)
    structure_weight = cfg.get("unpaired_structure_weight", 0.)
    air_weight = cfg.get("unpaired_air_weight", 0.)
    body_supervision = cfg.get("paired_reconstruction_mode", "full_fov") != "full_fov"
    gradient_weight = cfg.get("paired_gradient_weight", 0.)
    mse_weight = cfg.get("paired_mse_weight", 0.)
    perceptual = (
        VGGPerceptualLoss(
            cfg.get("vgg_layers", [3, 8, 17, 26, 35]), cfg.get("vgg_criterion", "l1"),
            slice_mode=cfg.get("vgg_slice_mode", "central"),
            feature_reduction=cfg.get("vgg_feature_reduction", "mean"),
            pooling=cfg.get("vgg_pooling", "max"),
            target_mode=cfg.get("vgg_target_mode", "spatial"),
        ).to(device)
        if perceptual_weight
        else None
    )
    if world > 1:
        args = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        generators = DistributedDataParallel(generators, **args)
        discriminators = DistributedDataParallel(discriminators, **args)
    generator_module = unwrap(generators)
    groups = [{"params": list(generator_module.mr_to_ct.parameters()) + list(generator_module.ct_to_mr.parameters()), "lr_ratio": 1.}]
    if generator_module.projection_heads is not None:
        ratio = cfg.get("projection_lr_multiplier", 1.)
        groups.append({"params": generator_module.projection_heads.parameters(),
                       "lr": cfg["lr"] * ratio, "lr_ratio": ratio})
    optim_g = torch.optim.Adam(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))
    optim_d = torch.optim.Adam(
        discriminators.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"])
    )
    precision = cfg.get("precision", "fp32")
    if precision not in {"fp32", "bf16", "fp16"} or (
        precision == "fp16" and device.type != "cuda"
    ):
        raise ValueError("Use fp32/bf16, or fp16 on CUDA")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    replay_mr, replay_ct = (
        ReplayBuffer(cfg.get("replay_size", 50), cfg["seed"] + rank),
        ReplayBuffer(cfg.get("replay_size", 50), cfg["seed"] + rank + 10000),
    )
    output.mkdir(parents=True, exist_ok=True)
    if state is not None:
        unwrap(generators).load_state_dict(state["generators"], strict=True)
        unwrap(discriminators).load_state_dict(state["discriminators"], strict=True)
        optim_g.load_state_dict(state["optim_g"])
        optim_d.load_state_dict(state["optim_d"])
        scaler.load_state_dict(state["scaler"])
        rank_filename = state.get(
            "rank_state_files", [f"state_rank{index}.pt" for index in range(world)]
        )[rank]
        if Path(rank_filename).name != rank_filename:
            raise ValueError("Rank checkpoint references must be local filenames")
        rank_state = torch.load(
            Path(resume).with_name(rank_filename),
            map_location="cpu",
            weights_only=True,
        )
        if (rank_state["epoch"] != state["next_epoch"]
                or rank_state.get("checkpoint_id") != state.get("checkpoint_id")):
            raise ValueError(
                "Main checkpoint and rank RNG/replay checkpoint belong to different epochs"
            )
        torch.set_rng_state(rank_state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rank_state["cuda_rng"], device)
        replay_mr.load_state_dict(rank_state["replay_mr"])
        replay_ct.load_state_dict(rank_state["replay_ct"])
        if replay_scope == "global" and rank != 0 and (replay_mr.items or replay_ct.items):
            raise ValueError("Only rank zero may hold saved global replay images")
    if rank == 0:
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if state is None:
            fingerprints = {}
            for name, network in (("generators", generators), ("discriminators", discriminators)):
                checksum = hashlib.sha256()
                for key, value in unwrap(network).state_dict().items():
                    checksum.update(key.encode())
                    checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
                fingerprints[name] = checksum.hexdigest()
            record = {
                "seed": cfg["seed"], "steps": 0,
                "rsam_initialization": cfg.get("initialization", "kaiming"),
                "pretrained_components": ["frozen ImageNet VGG19"] if perceptual is not None else [],
                "optimizer_state_entries": {"generator": len(optim_g.state), "discriminator": len(optim_d.state)},
                "replay_items": {"mr": len(replay_mr.items), "ct": len(replay_ct.items)},
                "parameter_sha256": fingerprints, "manifest_sha256": manifest_hash,
            }
            (output / "initialization_audit.json").write_text(json.dumps(record, indent=2) + "\n")
        # Record initialization only for a new training run.
        initial_provenance = {
            **ct_context_provenance,
            "steps": steps,
            "manifest_sha256": manifest_hash,
            "torch_version": str(torch.__version__),
            "world_size": world,
            "global_batch_size_max": batch_size * world,
            "precision": precision,
            "deterministic_algorithms": deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "epoch_complete": True,
            "status": "new_training_run",
            "sampler_version": active_sampler_version,
            "region_probabilities": sampler.region_probabilities,
            "replay_scope": replay_scope,
        }
        if save_interval and state is None:
            save_epoch_snapshot(output, unwrap(generators).mr_to_ct,
                                config, initial_provenance, 0)
        elif save_interval and epoch_start % save_interval == 0:
            # Export a missing snapshot at the current epoch boundary.
            path = output / "snapshots" / f"generator_mr_to_ct_epoch{epoch_start:04d}.pt"
            if not path.exists():
                initial_provenance["checkpoint_id"] = state["checkpoint_id"]
                save_epoch_snapshot(output, unwrap(generators).mr_to_ct,
                                    config, initial_provenance, epoch_start)
    if world > 1:
        dist.barrier()
    log = []
    last_completed_epoch = epoch_start
    for epoch in range(epoch_start, cfg["epochs"]):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        decay = cfg["decay_start_epoch"]
        multiplier = (
            1.0
            if epoch < decay
            else max(0.0, 1 - (epoch - decay) / max(1, cfg["epochs"] - decay))
        )
        for optimizer in (optim_g, optim_d):
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * multiplier * group.get("lr_ratio", 1.)
        # Weight loss averages by the number of samples.
        totals = torch.zeros(7, device=device, dtype=torch.float64)
        anchor_total = totals.new_zeros(())
        anatomy_totals = totals.new_zeros(2)
        gradient_total = totals.new_zeros(())
        mse_total = totals.new_zeros(())
        extra_totals = totals.new_zeros(2)
        region_counts = torch.zeros(len(config["region_order"]), device=device, dtype=torch.long)
        completed_batches = 0
        for batch in loader:
            mr, ct, code = (
                batch[name].to(device, non_blocking=True)
                for name in ["mr", "ct", "code"]
            )
            optim_g.zero_grad(set_to_none=True)
            unwrap(discriminators).requires_grad_(False)
            context = (
                torch.autocast(device.type, dtype=amp_dtype)
                if precision != "fp32"
                else nullcontext()
            )
            with context:
                generated = generators(mr, ct, code, step=steps, seed=cfg["seed"])
                d_mr, d_ct = unwrap(discriminators)(
                    generated["fake_mr"], ct_critic_input(generated["fake_ct"], batch, "fake", crop_indices)
                )
                weights = cfg["loss_weights"]
                loss_adv = (
                    adversarial_loss(d_mr, True, cfg["gan_loss"], aggregation=cfg.get("discriminator_aggregation", "patch"))
                    + adversarial_loss(d_ct, True, cfg["gan_loss"], aggregation=cfg.get("discriminator_aggregation", "patch"))
                )
                loss_cycle = (
                    F.l1_loss(generated["cycle_mr"], mr)
                    + F.l1_loss(generated["cycle_ct"], ct)
                )
                loss_identity = (
                    F.l1_loss(generated["identity_mr"], mr)
                    + F.l1_loss(generated["identity_ct"], ct)
                )
                loss_perceptual = loss_adv.new_zeros(())
                if perceptual is not None:
                    loss_perceptual = perceptual(
                        generated["fake_ct"], ct, paired=batch["paired"].to(device)
                    )
                    if cfg.get("perceptual_both_directions", False):
                        loss_perceptual = loss_perceptual + perceptual(
                            generated["fake_mr"], mr, paired=batch["paired"].to(device)
                        )
                loss_anchor = loss_adv.new_zeros(())
                loss_structure = loss_adv.new_zeros(())
                loss_air = loss_adv.new_zeros(())
                loss_gradient = loss_adv.new_zeros(())
                loss_mse = loss_adv.new_zeros(())
                loss_full_fov = loss_adv.new_zeros(())
                loss_nce = generated.get("nce_loss", loss_adv.new_zeros(()))
                if anchor_weight or structure_weight or air_weight or gradient_weight or mse_weight or full_fov_weight:
                    paired_mask = batch["paired"].to(device)
                    counts = torch.stack([paired_mask.sum(), (~paired_mask).sum()]).to(dtype=torch.float32).detach()
                    if world > 1:
                        dist.all_reduce(counts)
                    if anchor_weight:
                        if body_supervision:
                            loss_anchor = body_reconstruction_loss(
                                generated["fake_ct"], ct, paired_mask, batch["ct_body"].to(device),
                                background_weight=cfg.get("paired_background_weight", .05),
                                global_pair_count=counts[0], world_size=world,
                            )
                        else:
                            loss_anchor = paired_reconstruction_loss(
                                generated["fake_ct"], ct, paired_mask,
                                global_pair_count=counts[0], world_size=world,
                            )
                    if full_fov_weight:
                        loss_full_fov = paired_reconstruction_loss(
                            generated["fake_ct"], ct, paired_mask,
                            global_pair_count=counts[0], world_size=world,
                        )
                    if gradient_weight:
                        loss_gradient = paired_gradient_loss(
                            generated["fake_ct"], ct, paired_mask, batch["ct_body"].to(device),
                            global_pair_count=counts[0], world_size=world,
                        )
                    if mse_weight:
                        loss_mse = paired_foreground_mse_loss(
                            generated["fake_ct"], ct, paired_mask, batch["ct_body"].to(device),
                            global_pair_count=counts[0], world_size=world,
                        )
                    if structure_weight or air_weight:
                        loss_structure, loss_air = unpaired_anatomy_losses(
                            generated["fake_ct"], mr, ~paired_mask, batch["mr_body"].to(device),
                            global_unpaired_count=counts[1], world_size=world,
                        )
                loss_g = (
                    weights["adversarial"] * loss_adv
                    + weights["cycle"] * loss_cycle
                    + weights["identity"] * loss_identity
                    + perceptual_weight * loss_perceptual
                    + anchor_weight * loss_anchor
                    + structure_weight * loss_structure
                    + air_weight * loss_air
                    + gradient_weight * loss_gradient
                    + mse_weight * loss_mse
                    + full_fov_weight * loss_full_fov
                    + nce_weight * loss_nce
                )
            if not torch.isfinite(loss_g):
                raise FloatingPointError("Nonfinite generator loss")
            scaler.scale(loss_g).backward()
            scaler.step(optim_g)
            optim_d.zero_grad(set_to_none=True)
            unwrap(discriminators).requires_grad_(True)
            context = (
                torch.autocast(device.type, dtype=amp_dtype)
                if precision != "fp32"
                else nullcontext()
            )
            with context:
                fake_mr, fake_ct = (
                    distributed_replay_query(replay_mr, generated["fake_mr"], replay_scope),
                    distributed_replay_query(replay_ct, generated["fake_ct"], replay_scope),
                )
                real_a, fake_a, real_b, fake_b = discriminators(
                    mr, ct_critic_input(ct, batch, "real", crop_indices),
                    fake_mr, ct_critic_input(fake_ct, batch, "fake", crop_indices)
                )
                loss_d = 0.5 * (
                    adversarial_loss(real_a, True, cfg["gan_loss"], aggregation=cfg.get("discriminator_aggregation", "patch"))
                    + adversarial_loss(fake_a, False, cfg["gan_loss"], aggregation=cfg.get("discriminator_aggregation", "patch"))
                    + adversarial_loss(real_b, True, cfg["gan_loss"], aggregation=cfg.get("discriminator_aggregation", "patch"))
                    + adversarial_loss(fake_b, False, cfg["gan_loss"], aggregation=cfg.get("discriminator_aggregation", "patch"))
                )
            if not torch.isfinite(loss_d):
                raise FloatingPointError("Nonfinite discriminator loss")
            scaler.scale(loss_d).backward()
            scaler.step(optim_d)
            scaler.update()
            totals[:6] += torch.stack([
                loss_g.detach(), loss_d.detach(), loss_adv.detach(),
                loss_cycle.detach(), loss_identity.detach(), loss_perceptual.detach(),
            ]).double() * mr.shape[0]
            totals[6] += mr.shape[0]
            anchor_total += loss_anchor.detach().double() * mr.shape[0]
            gradient_total += loss_gradient.detach().double() * mr.shape[0]
            mse_total += loss_mse.detach().double() * mr.shape[0]
            extra_totals += torch.stack([loss_full_fov.detach(), loss_nce.detach()]).double() * mr.shape[0]
            anatomy_totals += torch.stack([loss_structure.detach(), loss_air.detach()]).double() * mr.shape[0]
            region_counts += code.sum(dim=0).long()
            completed_batches += 1
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if world > 1:
            dist.all_reduce(totals)
            dist.all_reduce(anchor_total)
            dist.all_reduce(anatomy_totals)
            dist.all_reduce(gradient_total)
            dist.all_reduce(mse_total)
            dist.all_reduce(extra_totals)
            dist.all_reduce(region_counts)
        elapsed = torch.tensor(time.perf_counter() - epoch_started, device=device)
        if world > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        epoch_complete = completed_batches == len(loader)
        checkpoint_id, rank_files = None, None
        if epoch_complete:
            last_completed_epoch = epoch + 1
            token = [uuid.uuid4().hex if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(token, src=0)
            checkpoint_id = token[0]
            rank_files = [
                f"state_epoch{epoch + 1:04d}_{checkpoint_id}_rank{index}.pt"
                for index in range(world)
            ]
            rank_state = {
                "epoch": epoch + 1,
                "checkpoint_id": checkpoint_id,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
                "replay_mr": replay_mr.state_dict(),
                "replay_ct": replay_ct.state_dict(),
            }
            # Commit rank state before the main checkpoint.
            atomic_save(rank_state, output / rank_files[rank])
        if world > 1:
            dist.barrier()
        if rank == 0:
            row = {
                **ct_context_provenance,
                "epoch": epoch,
                "steps": steps,
                "generator_loss": float(totals[0] / totals[6]),
                "discriminator_loss": float(totals[1] / totals[6]),
                "adversarial_loss": float(totals[2] / totals[6]),
                "cycle_loss": float(totals[3] / totals[6]),
                "identity_loss": float(totals[4] / totals[6]),
                "perceptual_loss": float(totals[5] / totals[6]),
                "perceptual_weight": perceptual_weight,
                "paired_reconstruction_loss": float(anchor_total / totals[6]),
                "paired_reconstruction_weight": anchor_weight,
                "paired_reconstruction_mode": cfg.get("paired_reconstruction_mode", "full_fov"),
                "paired_gradient_loss": float(gradient_total / totals[6]),
                "paired_gradient_weight": gradient_weight,
                "paired_mse_loss": float(mse_total / totals[6]),
                "paired_mse_weight": mse_weight,
                "paired_full_fov_loss": float(extra_totals[0] / totals[6]),
                "paired_full_fov_weight": full_fov_weight,
                "patch_nce_loss": float(extra_totals[1] / totals[6]),
                "patch_nce_weight": nce_weight,
                "unpaired_structure_loss": float(anatomy_totals[0] / totals[6]),
                "unpaired_structure_weight": structure_weight,
                "unpaired_air_loss": float(anatomy_totals[1] / totals[6]),
                "unpaired_air_weight": air_weight,
                "discriminator_aggregation": cfg.get("discriminator_aggregation", "patch"),
                "deterministic_algorithms": deterministic,
                "sampler_version": active_sampler_version,
                "replay_scope": replay_scope,
                "region_probabilities": sampler.region_probabilities,
                "region_samples_seen": dict(zip(config["region_order"], region_counts.tolist())),
                "lr": cfg["lr"] * multiplier,
                "epoch_seconds": float(elapsed),
                "samples_seen": int(totals[6]),
                "samples_per_second": float(totals[6] / elapsed),
                "world_size": world,
                "global_batch_size_max": batch_size * world,
                "epoch_complete": epoch_complete,
            }
            log.append(row)
            with open(output / "training.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            provenance = {
                **ct_context_provenance,
                "steps": steps,
                "epochs_completed": last_completed_epoch,
                "manifest_sha256": manifest_hash,
                "torch_version": str(torch.__version__),
                "world_size": world,
                "global_batch_size_max": batch_size * world,
                "precision": precision,
                "deterministic_algorithms": deterministic,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "epoch_complete": epoch_complete,
                "status": "new_training_run",
                "sampler_version": active_sampler_version,
                "region_probabilities": sampler.region_probabilities,
                "replay_scope": replay_scope,
            }
            save_generator(
                output / "generator_mr_to_ct.pt",
                unwrap(generators).mr_to_ct,
                config["preprocessing"],
                config["region_order"],
                provenance,
            )
            save_generator(
                output / "generator_ct_to_mr.pt",
                unwrap(generators).ct_to_mr,
                config["preprocessing"],
                config["region_order"],
                provenance,
            )
            if epoch_complete:
                atomic_save(
                    {
                        "generators": unwrap(generators).state_dict(),
                        "discriminators": unwrap(discriminators).state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": config,
                        "model_config": unwrap(generators).mr_to_ct.config.to_dict(),
                        "manifest_sha256": manifest_hash,
                        "world_size": world,
                        "sampler_version": active_sampler_version,
                        "replay_scope": replay_scope,
                        "checkpoint_id": checkpoint_id,
                        "rank_state_files": rank_files,
                        "next_epoch": epoch + 1,
                        "steps": steps,
                    },
                    output / "training_last.pt",
                )
                if save_interval and (epoch + 1) % save_interval == 0:
                    path = save_epoch_snapshot(
                        output, unwrap(generators).mr_to_ct, config,
                        {**provenance, "checkpoint_id": checkpoint_id}, epoch + 1,
                    )
                    print(json.dumps({"snapshot": str(path), "epochs_completed": epoch + 1}),
                          flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return log
