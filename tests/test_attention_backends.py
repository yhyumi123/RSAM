import copy

import pytest
import torch

from rsam.model import (
    AttResDecoder, Decoder, Generator, ModelConfig, ReflectionPad2d,
    RegionSpecificAttention,
)


@pytest.mark.parametrize("scaled", [True, False])
@pytest.mark.parametrize("control", ["full", "rav_neutral", "dgf_zero", "dgf_one"])
def test_sdpa_matches_eager_outputs_and_gradients(scaled, control):
    torch.set_num_threads(1)
    torch.manual_seed(128)
    config = ModelConfig(base_channels=8, image_size=32, hidden_dim=8,
                         attention_backend="eager", scaled_dot_product=scaled)
    eager = RegionSpecificAttention(32, config).double()
    sdpa = copy.deepcopy(eager)
    sdpa.attention_backend = "sdpa"
    x = torch.randn(2, 32, 8, 8, dtype=torch.double, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    codes = torch.eye(3, dtype=torch.double)[:2]
    expected = eager(x, codes, control)
    actual = sdpa(other, codes, control)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(other.grad, x.grad, atol=1e-12, rtol=1e-12)
    for left, right in zip(eager.parameters(), sdpa.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-12, rtol=1e-12)


def test_sdpa_trace_materializes_only_requested_attention_rows(monkeypatch):
    torch.set_num_threads(1)
    model = RegionSpecificAttention(32, ModelConfig(image_size=32, hidden_dim=8))
    x, code = torch.randn(1, 32, 8, 8), torch.eye(3)[:1]
    original_bmm, shapes = torch.bmm, []

    def record_bmm(a, b):
        shapes.append((a.shape, b.shape))
        return original_bmm(a, b)

    monkeypatch.setattr(torch, "bmm", record_bmm)
    output, trace = model(x, code, query_points=[(1, 2), (3, 4)])
    assert trace["attention_with_rav"].shape == (1, 2, 64)
    assert not trace["attention_with_rav"].requires_grad
    assert shapes and all(a[-2] == 2 for a, _ in shapes)
    torch.testing.assert_close(output, model(x, code), rtol=0, atol=0)
    eager = copy.deepcopy(model)
    eager.attention_backend = "eager"
    _, reference = eager(x, code, query_points=[(1, 2), (3, 4)])
    torch.testing.assert_close(trace["attention_with_rav"], reference["attention_with_rav"])


@pytest.mark.parametrize("override", [
    {"base_channels": 0}, {"input_nc": 2}, {"image_size": 31},
    {"attention_blocks": [1, 1]}, {"attention_blocks": [0]},
    {"attention_backend": "invalid"}, {"scaled_dot_product": "true"},
])
def test_model_rejects_invalid_configuration(override):
    with pytest.raises(ValueError):
        ModelConfig(**override)


def test_attresdecoder_is_complete_default_decoder():
    model = Generator(ModelConfig(base_channels=8, image_size=32))
    assert isinstance(model.decoder, AttResDecoder)
    assert Decoder is AttResDecoder
    assert [i + 1 for i, block in enumerate(model.decoder.res_blocks)
            if hasattr(block, "attention")] == [1, 3]


@pytest.mark.parametrize("padding", [1, 3, (1, 2, 3, 1), (-1, 2, 0, 1)])
def test_deterministic_reflection_preserves_outputs_and_gradients(padding):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        x = torch.randn(2, 3, 8, 9, dtype=torch.double, requires_grad=True)
        expected = torch.nn.ReflectionPad2d(padding)(x)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        other = x.detach().clone().requires_grad_(True)
        torch.use_deterministic_algorithms(True)
        actual = ReflectionPad2d(padding)(other)
        actual.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(other.grad, x.grad, rtol=1e-12, atol=1e-12)
        assert ReflectionPad2d(padding).state_dict() == {}
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
