"""Adversarial, cycle, identity, and frozen VGG19 perceptual objectives."""

import random

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


def adversarial_loss(logits, real, kind, aggregation="patch"):
    if aggregation not in {"patch", "global_mean"}:
        raise ValueError("aggregation must be patch or global_mean")
    if aggregation == "global_mean":
        if logits.ndim != 4:
            raise ValueError("global_mean expects BCHW discriminator logits")
        logits = logits.mean((1, 2, 3), keepdim=True)
    target = torch.ones_like(logits) if real else torch.zeros_like(logits)
    if kind == "lsgan":
        return F.mse_loss(logits, target)
    if kind == "logistic":
        return F.binary_cross_entropy_with_logits(logits, target)
    raise ValueError("gan_loss must be logistic or lsgan")


def paired_reconstruction_loss(predicted, target, paired, global_pair_count=None, world_size=1):
    """Paired CT-only L1; DDP gradients equal a global paired-example mean."""
    if (predicted.ndim != 4 or predicted.shape != target.shape
            or not isinstance(paired, torch.Tensor) or paired.dtype != torch.bool
            or paired.shape != (predicted.shape[0],) or paired.device != predicted.device):
        raise ValueError("paired reconstruction requires matching BCHW and a boolean B mask")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    count = paired.sum() if global_pair_count is None else global_pair_count
    count = torch.as_tensor(count, device=predicted.device, dtype=predicted.dtype).detach()
    if count.ndim != 0 or not torch.isfinite(count) or count < paired.sum():
        raise ValueError("global_pair_count must cover local paired examples")
    per_case = (predicted - target.detach()).abs().mean((1, 2, 3))
    return (per_case * paired).sum() * world_size / count.clamp_min(1)


def validate_vgg_target_mode(mode, criterion="l1", reduction="mean"):
    if mode not in {"spatial", "paired_spatial_unpaired_statistics"}:
        raise ValueError("Unknown VGG target mode")
    if mode != "spatial" and (criterion != "l1" or reduction not in {"mean", "channel_sum"}):
        raise ValueError("Statistics VGG targets require L1 and mean or channel_sum reduction")


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 17, 26, 35), criterion="l1",
                 slice_mode="central", feature_reduction="mean", pooling="max",
                 target_mode="spatial"):
        super().__init__()
        layers = tuple(layers)
        if (not layers or any(type(index) is not int or not 0 <= index < 37
                              for index in layers)):
            raise ValueError("VGG layers must be nonempty feature indices between 0 and 36")
        if criterion not in {"l1", "mse"}:
            raise ValueError("VGG criterion must be l1 or mse")
        if slice_mode not in {"central", "all", "stack"}:
            raise ValueError("slice_mode must be central, all or stack")
        if pooling not in {"max", "avg"}:
            raise ValueError("VGG pooling must be max or avg")
        if feature_reduction not in {"mean", "channel_sum", "sample_sum"}:
            raise ValueError("feature_reduction must be mean, channel_sum or sample_sum")
        validate_vgg_target_mode(target_mode, criterion, feature_reduction)
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        if pooling == "avg":
            # Use average pooling for the VGG features.
            for index, layer in enumerate(self.features):
                if isinstance(layer, nn.MaxPool2d):
                    self.features[index] = nn.AvgPool2d(2, stride=2, padding=0)
        self.features.requires_grad_(False)
        self.layers = set(layers)
        self.criterion = criterion
        self.slice_mode = slice_mode
        self.feature_reduction = feature_reduction
        self.pooling = pooling
        self.target_mode = target_mode
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, predicted, target, paired=None):
        if (predicted.ndim != 4 or predicted.shape != target.shape
                or predicted.shape[1] % 2 != 1
                or min(predicted.shape[-2:]) < 32
                or predicted.device != target.device):
            raise ValueError("VGG inputs require matching BCHW odd-channel stacks, size >= 32")
        if self.target_mode != "spatial":
            if (not isinstance(paired, torch.Tensor) or paired.dtype != torch.bool
                    or paired.shape != (predicted.shape[0],) or paired.device != predicted.device):
                raise ValueError("Statistics VGG targets require a boolean paired mask of shape [B] on the input device")
        target = target.detach()
        if self.slice_mode == "stack":
            if predicted.shape[1] != 3:
                raise ValueError("stack VGG inputs require exactly three adjacent slices")
            p, t = (predicted + 1) / 2, (target + 1) / 2
        elif self.slice_mode == "all":
            p = predicted.flatten(0, 1).unsqueeze(1)
            t = target.flatten(0, 1).unsqueeze(1)
            if self.target_mode != "spatial":
                paired = paired.repeat_interleave(predicted.shape[1])
        else:
            middle = predicted.shape[1] // 2
            p = predicted[:, middle:middle+1]
            t = target[:, middle:middle+1]
        if self.slice_mode != "stack":
            p, t = ((p + 1) / 2).repeat(1, 3, 1, 1), ((t + 1) / 2).repeat(1, 3, 1, 1)
        p, t = (p - self.mean) / self.std, (t - self.mean) / self.std
        total = p.new_zeros(())
        for index, layer in enumerate(self.features):
            p = layer(p)
            with torch.no_grad():
                t = layer(t)
            if index in self.layers:
                if self.target_mode != "spatial":
                    # Use spatial features for paired targets and moments for unpaired targets.
                    spatial = (p - t).abs().mean((-2, -1))
                    # Clone feature planes for standard-deviation gradients.
                    p_flat, t_flat = p.flatten(2).clone(), t.flatten(2)
                    moments = 0.5 * (
                        (p_flat.mean(-1) - t_flat.mean(-1)).abs()
                        + (p_flat.std(-1, unbiased=False) - t_flat.std(-1, unbiased=False)).abs()
                    )
                    per_channel = torch.where(paired[:, None], spatial, moments)
                    total = total + (per_channel.mean() if self.feature_reduction == "mean"
                                     else per_channel.sum(1).mean())
                elif self.feature_reduction in {"channel_sum", "sample_sum"}:
                    error = (p-t).abs() if self.criterion == "l1" else (p-t).square()
                    if self.feature_reduction == "sample_sum":
                        total = total + error.flatten(1).sum(1).mean()
                    else:
                        total = total + error.sum(1).mean()
                else:
                    total = total + (
                        F.l1_loss(p, t) if self.criterion == "l1" else F.mse_loss(p, t)
                    )
            if index >= max(self.layers):
                break
        return total


class ReplayBuffer:
    def __init__(self, capacity=50, seed=0):
        self.capacity = capacity
        self.items = []
        self.rng = random.Random(seed)

    def query(self, batch):
        result = []
        for image in batch.detach():
            if self.capacity <= 0:
                result.append(image)
            elif len(self.items) < self.capacity:
                self.items.append(image.cpu().clone())
                result.append(image)
            elif self.rng.random() > 0.5:
                index = self.rng.randrange(self.capacity)
                result.append(self.items[index].to(batch.device))
                self.items[index] = image.cpu().clone()
            else:
                result.append(image)
        return torch.stack(result)

    def state_dict(self):
        return {"items": self.items, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.items = state["items"]
        self.rng.setstate(state["rng"])


def distributed_replay_query(buffer, batch, scope="per_rank"):
    """Query a synchronized replay pool or independent rank-local pools."""
    if scope not in {"per_rank", "global"}:
        raise ValueError("replay_scope must be per_rank or global")
    distributed = dist.is_available() and dist.is_initialized()
    if scope == "per_rank" or not distributed or dist.get_world_size() == 1:
        return buffer.query(batch)
    batch = batch.detach().contiguous()
    if batch.ndim < 1 or batch.shape[0] < 1:
        raise ValueError("Global replay requires a nonempty local batch")
    rank, world = dist.get_rank(), dist.get_world_size()
    gathered = [torch.empty_like(batch) for _ in range(world)] if rank == 0 else None
    dist.gather(batch, gather_list=gathered, dst=0)
    selected_by_rank = None
    if rank == 0:
        global_batch = torch.stack(gathered, dim=1).flatten(0, 1)
        selected = buffer.query(global_batch).reshape(batch.shape[0], world, *batch.shape[1:])
        selected_by_rank = [part.contiguous() for part in selected.unbind(dim=1)]
    result = torch.empty_like(batch)
    dist.scatter(result, scatter_list=selected_by_rank, src=0)
    return result
