"""VGG feature extraction and L1 perceptual loss."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchvision.models import vgg19

from rsam import losses


RELU_LAYERS = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35]


def test_stack_matches_independent_full_feature_forward_and_gradient(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(438)
    features = vgg19(weights=None).features.eval()
    monkeypatch.setattr(losses, 'vgg19', lambda weights: SimpleNamespace(features=deepcopy(features)))
    objective = losses.VGGPerceptualLoss(
        layers=RELU_LAYERS, criterion='l1', slice_mode='stack', feature_reduction='mean', pooling='avg')

    # Check selected VGG features with an independent forward pass.
    oracle = deepcopy(features).requires_grad_(False)
    for index, layer in enumerate(oracle):
        if isinstance(layer, nn.MaxPool2d):
            oracle[index] = nn.AvgPool2d(2, 2, 0)
    activations = []
    handles = [layer.register_forward_hook(lambda _m, _i, out: activations.append(out.clone()))
               for layer in oracle if isinstance(layer, nn.ReLU)]
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    target = (torch.rand_like(x) * 2 - 1).requires_grad_()
    predicted = x.clone().requires_grad_()
    actual = objective(predicted, target)
    actual_gradient, = torch.autograd.grad(actual, predicted)
    p = x.clone().requires_grad_()
    mean = p.new_tensor([.485, .456, .406])[None, :, None, None]
    std = p.new_tensor([.229, .224, .225])[None, :, None, None]
    oracle(((p + 1) / 2 - mean) / std)
    generated_features = list(activations)
    activations.clear()
    with torch.no_grad():
        oracle(((target + 1) / 2 - mean) / std)
    expected = torch.stack([(a - b).abs().mean((1, 2, 3))
                            for a, b in zip(generated_features, activations)]).sum(0).mean()
    expected_gradient, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7)
    assert len(activations) == len(RELU_LAYERS) == 16
    assert (actual_gradient.abs().sum((0, 2, 3)) > 0).all()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in objective.features.parameters())
    assert target.grad is None
    for handle in handles:
        handle.remove()


def test_stack_retains_three_distinct_inputs_and_spatial_mean(monkeypatch):
    monkeypatch.setattr(losses, 'vgg19', lambda weights: SimpleNamespace(features=nn.Sequential(nn.Identity())))
    objective = losses.VGGPerceptualLoss(layers=[0], slice_mode='stack', pooling='avg')
    seen = []
    handle = objective.features[0].register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].detach().clone()))
    p = torch.tensor([-.8, .1, .6])[None, :, None, None].expand(2, 3, 32, 32)
    t = torch.zeros_like(p)
    small = objective(p, t)
    normalized = ((p + 1) / 2 - objective.mean) / objective.std
    torch.testing.assert_close(seen[0], normalized, rtol=0, atol=0)
    assert seen[0].shape == (2, 3, 32, 32)
    large = objective(p.repeat_interleave(2, -1).repeat_interleave(2, -2),
                      t.repeat_interleave(2, -1).repeat_interleave(2, -2))
    torch.testing.assert_close(small, large)
    handle.remove()
    with pytest.raises(ValueError, match='exactly three'):
        objective(p[:, :1], t[:, :1])


def test_pooling_choice_is_explicit(monkeypatch):
    monkeypatch.setattr(losses, 'vgg19', lambda weights: SimpleNamespace(features=nn.Sequential(nn.MaxPool2d(2))))
    assert isinstance(losses.VGGPerceptualLoss(layers=[0]).features[0], nn.MaxPool2d)
    assert isinstance(losses.VGGPerceptualLoss(layers=[0], pooling='avg').features[0], nn.AvgPool2d)
    with pytest.raises(ValueError, match='pooling'):
        losses.VGGPerceptualLoss(pooling='invalid')
