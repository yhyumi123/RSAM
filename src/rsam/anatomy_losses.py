"""Paired reconstruction and unpaired structural losses."""
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage


def _validate_body_envelope_settings(component_mode, min_component_pixels,
                                     relative_component_size):
    if component_mode not in ('largest', 'substantial_components'):
        raise ValueError('body_envelope_mode must be largest or substantial_components')
    if (isinstance(min_component_pixels, (bool, np.bool_))
            or not isinstance(min_component_pixels, Integral) or min_component_pixels < 1):
        raise ValueError('body_envelope_min_component_pixels must be a positive integer')
    if (isinstance(relative_component_size, (bool, np.bool_))
            or not isinstance(relative_component_size, Real)
            or not np.isfinite(relative_component_size)
            or not 0 <= relative_component_size <= 1):
        raise ValueError('body_envelope_relative_component_size must be finite and in [0, 1]')


def validate_body_envelope_config(training):
    """Validate body-mask settings."""
    if not isinstance(training, Mapping):
        raise ValueError('Body envelope settings require a training mapping')
    options = {
        'component_mode': training.get('body_envelope_mode', 'largest'),
        'min_component_pixels': training.get('body_envelope_min_component_pixels', 64),
        'relative_component_size': training.get('body_envelope_relative_component_size', .01),
    }
    _validate_body_envelope_settings(**options)
    return options


def body_envelope(stack, threshold, *, component_mode='largest',
                  min_component_pixels=64, relative_component_size=.01):
    """Build a training body mask from selected connected components."""
    _validate_body_envelope_settings(component_mode, min_component_pixels,
                                     relative_component_size)
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not np.isfinite(threshold)):
        raise ValueError('Body envelope threshold must be finite and real')
    if (not isinstance(stack, torch.Tensor) or stack.device.type != 'cpu'
            or stack.layout != torch.strided or stack.ndim != 3
            or not all(stack.shape) or stack.is_complex()
            or not torch.isfinite(stack).all()):
        raise ValueError('Body envelope requires a finite CPU CHW stack')
    values = stack.detach()
    if values.dtype == torch.bfloat16:
        values = values.float()
    masks = []
    for image in values.numpy():
        foreground = image > threshold
        foreground = foreground | ndimage.binary_closing(foreground, iterations=2)
        labels, count = ndimage.label(foreground)
        if not count:
            masks.append(np.zeros_like(foreground))
            continue
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        largest = sizes.argmax()
        if component_mode == 'largest':
            mask = ndimage.binary_fill_holes(labels == largest)
        else:
            cutoff = max(min_component_pixels, relative_component_size * sizes[largest])
            retained = np.flatnonzero(sizes >= cutoff)
            retained = retained[retained != 0]
            retained = np.union1d(retained, [largest])
            mask = np.zeros_like(foreground)
            for component in retained:
                mask |= ndimage.binary_fill_holes(labels == component)
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)).bool()


def _validate(predicted, target, selected, body, count, world_size):
    if (predicted.ndim != 4 or predicted.shape != target.shape
            or target.device != predicted.device or body.shape != predicted.shape
            or body.dtype != torch.bool or body.device != predicted.device
            or selected.shape != (predicted.shape[0],) or selected.dtype != torch.bool
            or selected.device != predicted.device):
        raise ValueError('Anatomy objectives require matching BCHW and Boolean masks')
    if type(world_size) is not int or world_size < 1:
        raise ValueError('world_size must be a positive integer')
    count = selected.sum() if count is None else count
    count = torch.as_tensor(count, device=predicted.device, dtype=predicted.dtype).detach()
    if count.ndim != 0 or not torch.isfinite(count) or count < selected.sum():
        raise ValueError('Global selected-example count must cover local examples')
    return count


def _spatial_mean(error, mask):
    return (error * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def body_reconstruction_loss(predicted, target, paired, body,
                             global_pair_count=None, world_size=1,
                             background_weight=0.05):
    count = _validate(predicted, target, paired, body, global_pair_count, world_size)
    error = (predicted - target.detach()).abs()
    per_case = _spatial_mean(error, body) + background_weight * _spatial_mean(error, ~body)
    per_case = torch.where(body.flatten(1).any(1), per_case, error.flatten(1).mean(1))
    return (per_case * paired).sum() * world_size / count.clamp_min(1)


def mind2d_descriptor(image):
    """Compute a slice-wise local self-similarity descriptor."""
    if image.ndim != 4 or min(image.shape[-2:]) < 5:
        raise ValueError('Descriptor requires BCHW spatial dimensions >=5')
    batch, channels, height, width = image.shape
    value = image.reshape(batch * channels, 1, height, width)
    offsets = [(-2, -2), (-2, 0), (-2, 2), (0, -2),
               (0, 2), (2, -2), (2, 0), (2, 2)]
    kernel = image.new_zeros((8, 1, 5, 5))
    for index, (dy, dx) in enumerate(offsets):
        kernel[index, 0, 2 + dy, 2 + dx] = 1
    neighbours = F.conv2d(F.pad(value, (2, 2, 2, 2), mode='replicate'), kernel)
    distance = F.avg_pool2d((neighbours - value).square(), 3, stride=1,
                            padding=1, count_include_pad=False)
    distance = distance - distance.amin(1, keepdim=True)
    variance = distance.mean(1, keepdim=True)
    floor = variance.mean((-2, -1), keepdim=True).detach() * 1e-3
    descriptor = torch.exp(-distance / torch.maximum(variance, floor).clamp_min(1e-6))
    return descriptor.reshape(batch, channels, 8, height, width)


def unpaired_anatomy_losses(predicted, mr, unpaired, body,
                            global_unpaired_count=None, world_size=1):
    count = _validate(predicted, mr, unpaired, body, global_unpaired_count, world_size)
    if not unpaired.any():
        zero = predicted.sum() * 0
        return zero, zero
    value, reference, inside = predicted[unpaired], mr[unpaired].detach(), body[unpaired]
    descriptor = mind2d_descriptor(value)
    with torch.no_grad():
        target_descriptor = mind2d_descriptor(reference)
    per_case = _spatial_mean((descriptor - target_descriptor).abs().mean(2), inside)
    air_per_case = _spatial_mean((value + 1).abs(), ~inside)
    scale = world_size / count.clamp_min(1)
    return per_case.sum() * scale, air_per_case.sum() * scale


def paired_gradient_loss(predicted, target, paired, body,
                         global_pair_count=None, world_size=1):
    """Compute multiscale CT gradient loss along body edges."""
    count = _validate(predicted, target, paired, body, global_pair_count, world_size)
    if min(predicted.shape[-2:]) < 4:
        raise ValueError('Two-scale CT gradients require spatial dimensions >=4')
    per_case = predicted.flatten(1).sum(1) * 0
    for scale in (1, 2):
        value, reference, inside = predicted, target.detach(), body
        if scale > 1:
            value = F.avg_pool2d(value, scale)
            reference = F.avg_pool2d(reference, scale)
            inside = F.avg_pool2d(inside.float(), scale) > 0
        dx = (value[..., 1:] - value[..., :-1]) - (reference[..., 1:] - reference[..., :-1])
        dy = (value[..., 1:, :] - value[..., :-1, :]) - (reference[..., 1:, :] - reference[..., :-1, :])
        mask_x = inside[..., 1:] | inside[..., :-1]
        mask_y = inside[..., 1:, :] | inside[..., :-1, :]
        per_case = per_case + .25 * (_spatial_mean(dx.abs(), mask_x) + _spatial_mean(dy.abs(), mask_y))
    return (per_case * paired).sum() * world_size / count.clamp_min(1)


def paired_foreground_mse_loss(predicted, target, paired, body,
                               global_pair_count=None, world_size=1):
    """Compute body-normalized paired MSE with DDP aggregation."""
    count = _validate(predicted, target, paired, body, global_pair_count, world_size)
    per_case = _spatial_mean((predicted - target.detach()).square(), body)
    return (per_case * paired).sum() * world_size / count.clamp_min(1)
