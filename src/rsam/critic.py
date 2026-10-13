"""Differentiable training inputs for the CT discriminator."""
import torch
from torch.nn import functional as F


def ct_critic_input(image, batch, role, region_indices=()):
    """Crop unpaired CT critic inputs to body support during training."""
    if role not in {"real", "fake"}:
        raise ValueError("role must be real or fake")
    if not region_indices:
        return image
    selected = (~batch["paired"].to(image.device).bool()
                & batch["code"].to(image.device)[:, list(region_indices)].bool().any(1))
    mask = batch["ct_body" if role == "real" else "mr_body"].to(image.device).bool()
    if mask.shape != image.shape:
        raise ValueError("Body mask must match the CT input")
    samples = []
    height, width = image.shape[-2:]
    for i in range(len(image)):
        if not bool(selected[i]):
            samples.append(image[i])
            continue
        value = torch.where(mask[i:i+1], image[i:i+1], -1.)
        points = mask[i].any(0).nonzero()
        if points.numel():
            y0, x0 = (points.min(0).values - 4).clamp(min=0).tolist()
            y1, x1 = (points.max(0).values + 5).tolist()
            value = F.interpolate(value[:, :, y0:min(y1,height), x0:min(x1,width)],
                                  size=(height,width), mode="bilinear",
                                  align_corners=False, antialias=True)
        samples.append(value[0])
    return torch.stack(samples)
