import torch

from rsam.checkpoint import load_generator, save_generator
from rsam.model import Generator, ModelConfig, RegionSpecificAttention


def small_config():
    return ModelConfig(
        input_nc=3,
        output_nc=3,
        base_channels=8,
        image_size=32,
        hidden_dim=8,
        n_residual_blocks=3,
        attention_blocks=(1, 3),
    )


def test_rav_registered_and_updated_by_optimizer():
    torch.manual_seed(12)
    module = RegionSpecificAttention(32, small_config())
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    parameter = module.mlp[2].weight
    before = parameter.detach().clone()
    ids = {id(p) for p in module.parameters()}
    x = torch.randn(2, 32, 8, 8)
    codes = torch.eye(3)[:2]
    module(x, codes).square().mean().backward()
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert module.mlp[2].weight is parameter
    assert ids == {id(p) for p in module.parameters()}
    assert not torch.equal(before, parameter)


def test_fixed_query_attention_and_gate_interventions():
    torch.manual_seed(13)
    module = RegionSpecificAttention(32, small_config()).eval()
    x, codes = torch.randn(1, 32, 8, 8), torch.tensor([[1.0, 0.0, 0.0]])
    with torch.no_grad():
        full, trace = module(x, codes, query_points=[(2, 3)])
        assert torch.equal(module(x, codes, "dgf_zero"), x)
        open_gate = module(x, codes, "dgf_one")
        gate = trace["gate"].view(1, 1, 1, 1)
        torch.testing.assert_close(full, x + gate * (open_gate - x))
        attention = trace["attention_with_rav"]
        torch.testing.assert_close(attention.sum(-1), torch.ones(1, 1))
        q = module.query_conv(x).flatten(2).transpose(1, 2)
        k = module.key_conv(x).flatten(2)
        reference = (torch.bmm(q[:, [2 * 8 + 3]], k) / 2).softmax(-1)
        torch.testing.assert_close(trace["attention_without_rav"], reference)
        assert torch.equal(full, module(x, codes))


def test_checkpoint_round_trip_2p5d(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    model = Generator(small_config()).eval()
    x, codes = torch.randn(1, 3, 32, 32), torch.eye(3)[1:2]
    path = tmp_path / "generator.pt"
    save_generator(
        path, model, {"ct_clip": [-1000, 2000]}, ["brain", "pelvis", "abdomen"]
    )
    loaded, _ = load_generator(path)
    with torch.no_grad():
        torch.testing.assert_close(model(x, codes), loaded(x, codes), rtol=0, atol=0)
