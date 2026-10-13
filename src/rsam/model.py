"""Region-specific attention with registered RAV and DGF parameters."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from rsam.normalization import instance_norm


@dataclass
class ModelConfig:
    input_nc: int = 3
    output_nc: int = 3
    num_regions: int = 3
    base_channels: int = 64
    n_residual_blocks: int = 9
    attention_blocks: tuple = (1, 3)
    hidden_dim: int = 128
    image_size: int = 256
    rav_activation: str = "identity"
    scaled_dot_product: bool = True
    attention_backend: str = "sdpa"
    rsam_second_relu: bool = True
    stable_instance_norm: bool = False
    attention_control: str = "full"

    def __post_init__(self):
        for name in (
            "input_nc", "output_nc", "num_regions", "base_channels",
            "n_residual_blocks", "hidden_dim", "image_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_nc % 2 != 1 or self.output_nc % 2 != 1:
            raise ValueError("Input and output channels must be odd for slice stacks")
        self.attention_blocks = tuple(self.attention_blocks)
        if self.image_size % 4 or self.image_size < 32:
            raise ValueError("image_size must be a multiple of 4 and at least 32")
        if self.base_channels % 2:
            raise ValueError("base_channels must be even")
        if (
            any(type(index) is not int for index in self.attention_blocks)
            or len(set(self.attention_blocks)) != len(self.attention_blocks)
            or not set(self.attention_blocks).issubset(
            range(1, self.n_residual_blocks + 1)
            )
        ):
            raise ValueError("attention_blocks require unique one-based residual block indices")
        if self.rav_activation not in {"identity", "relu"}:
            raise ValueError("rav_activation must be identity or relu")
        if type(self.scaled_dot_product) is not bool:
            raise ValueError("scaled_dot_product must be a boolean")
        if type(self.rsam_second_relu) is not bool:
            raise ValueError("rsam_second_relu must be a boolean")
        if type(self.stable_instance_norm) is not bool:
            raise ValueError("stable_instance_norm must be a boolean")
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be sdpa or eager")
        if (not isinstance(self.attention_control, str) or self.attention_control not in
                {"full", "rav_neutral", "dgf_one", "rav_neutral_dgf_one"}):
            raise ValueError("attention_control requires a supported attention configuration")

    def to_dict(self):
        result = asdict(self)
        result["attention_blocks"] = list(self.attention_blocks)
        return result


class RegionSpecificAttention(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        if type(channels) is not int or channels < 8 or channels % 8:
            raise ValueError("Attention channels must be a positive multiple of eight")
        self.channels = channels
        self.grid_size = config.image_size // 4
        self.scaled_dot_product = config.scaled_dot_product
        self.attention_backend = config.attention_backend
        self.num_regions = config.num_regions
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.mlp = nn.Sequential(
            nn.Linear(config.num_regions, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.grid_size**2),
            nn.ReLU() if config.rav_activation == "relu" else nn.Identity(),
        )
        self.gamma_mlp = nn.Sequential(
            nn.Linear(config.num_regions, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.attention_weight = nn.Parameter(torch.tensor([0.1]))
        # Freeze parameters for disabled regional modulation.
        if config.attention_control in {"rav_neutral", "rav_neutral_dgf_one"}:
            self.mlp.requires_grad_(False)
            self.attention_weight.requires_grad_(False)
        if config.attention_control in {"dgf_one", "rav_neutral_dgf_one"}:
            self.gamma_mlp.requires_grad_(False)

    def region_responses(self, codes):
        scale = 1 + self.attention_weight * self.mlp(codes)
        gate = self.gamma_mlp(codes)
        return scale, gate

    def forward(self, x, codes, control="full", query_points=None):
        if x.ndim != 4 or not x.is_floating_point() or x.shape[0] < 1:
            raise ValueError("Attention input must be a nonempty floating BCHW tensor")
        batch, channels, height, width = x.shape
        if channels != self.channels or (height, width) != (self.grid_size, self.grid_size):
            raise ValueError(
                "Feature grid disagrees with the registered RAV projection"
            )
        if (codes.shape != (batch, self.num_regions) or codes.device != x.device
                or not codes.is_floating_point()):
            raise ValueError(
                "Region codes must have shape [B, num_regions] on the input device"
            )
        if control not in {
            "full",
            "rav_neutral",
            "dgf_zero",
            "dgf_one",
            "regional_mean_gate",
            "rav_neutral_dgf_one",
        }:
            raise ValueError(f"Unknown intervention: {control}")
        q = self.query_conv(x).flatten(2).transpose(1, 2)
        k = self.key_conv(x).flatten(2)
        v = self.value_conv(x).flatten(2)
        scale, gate = self.region_responses(codes)
        applied_scale = (torch.ones_like(scale)
                         if control in {"rav_neutral", "rav_neutral_dgf_one"} else scale)
        if control == "dgf_zero":
            gate = torch.zeros_like(gate)
        elif control in {"dgf_one", "rav_neutral_dgf_one"}:
            gate = torch.ones_like(gate)
        elif control == "regional_mean_gate":
            all_codes = torch.eye(
                self.num_regions, dtype=codes.dtype, device=codes.device
            )
            gate = torch.ones_like(gate) * self.gamma_mlp(all_codes).mean()
        factor = 1 / math.sqrt(q.shape[-1]) if self.scaled_dot_product else 1.0
        modulated_q = q * applied_scale.unsqueeze(-1)
        if self.attention_backend == "sdpa":
            # Compute single-head attention with the configured scale.
            enhanced = F.scaled_dot_product_attention(
                modulated_q.unsqueeze(1).contiguous(),
                k.transpose(1, 2).unsqueeze(1).contiguous(),
                v.transpose(1, 2).unsqueeze(1).contiguous(),
                dropout_p=0.0, scale=factor,
            ).squeeze(1).transpose(1, 2).reshape_as(x)
        else:
            attention = (torch.bmm(modulated_q, k) * factor).softmax(-1)
            enhanced = torch.bmm(v, attention.transpose(1, 2)).reshape_as(x)
        update = gate[:, :, None, None] * enhanced
        y = x + update
        if query_points is None:
            return y
        indices = []
        for iy, ix in query_points:
            if (type(iy) is not int or type(ix) is not int
                    or not (0 <= iy < height and 0 <= ix < width)):
                raise ValueError("Query outside feature grid")
            indices.append(iy * width + ix)
        # Compute attention rows for the requested query positions.
        with torch.no_grad():
            unmodulated = (torch.bmm(q[:, indices], k) * factor).softmax(-1)
            selected = (torch.bmm(modulated_q[:, indices], k) * factor).softmax(-1)
        trace = {
            "attention_without_rav": unmodulated.detach(),
            "attention_with_rav": selected,
            "scale": scale.reshape(batch, height, width).detach(),
            "gate": gate.detach(),
            "update_norm": update.float().square().sum(1).sqrt().detach(),
            "eta": (
                update.float().flatten(1).norm(dim=1)
                / (x.float().flatten(1).norm(dim=1) + 1e-12)
            ).detach(),
            "query_points_yx": query_points,
            "grid_shape": (height, width),
        }
        return y, trace


class ReflectionPad2d(nn.ReflectionPad2d):
    """Reflection padding with deterministic backward when enabled."""

    def forward(self, x):
        if not torch.are_deterministic_algorithms_enabled():
            return super().forward(x)
        if x.ndim not in {3, 4}:
            raise ValueError("ReflectionPad2d expects CHW or BCHW input")
        left, right, top, bottom = self.padding

        def indices(size, before, after):
            if max(before, after) >= size or size + before + after <= 0:
                raise ValueError("Reflection padding must be smaller than the input dimension")
            values = torch.arange(-before, size + after, device=x.device)
            values = torch.where(values < 0, -values, values)
            return torch.where(values >= size, 2 * (size - 1) - values, values)

        return x.index_select(-2, indices(x.shape[-2], top, bottom)).index_select(
            -1, indices(x.shape[-1], left, right)
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels, config, with_attention):
        super().__init__()
        self.conv_block = nn.Sequential(
            ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            instance_norm(channels, config.stable_instance_norm),
            nn.ReLU(inplace=True),
            ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            instance_norm(channels, config.stable_instance_norm),
        )
        if with_attention:
            if config.rsam_second_relu:
                self.conv_block.append(nn.ReLU(inplace=True))
            self.attention = RegionSpecificAttention(channels, config)

    def forward(self, x, codes, control="full", query_points=None):
        out = self.conv_block(x)
        trace = None
        if hasattr(self, "attention"):
            out = self.attention(out, codes, control, query_points)
            if query_points is not None:
                out, trace = out
        return x + out, trace


class AttResDecoder(nn.Module):
    """Residual decoder with regional attention."""
    def __init__(self, config):
        super().__init__()
        channels = config.base_channels * 4
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, config, i + 1 in config.attention_blocks)
                for i in range(config.n_residual_blocks)
            ]
        )
        layers = []
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels, channels // 2, 3, 2, 1, output_padding=1
                    ),
                    instance_norm(channels // 2, config.stable_instance_norm),
                    nn.ReLU(inplace=True),
                ]
            )
            channels //= 2
        layers.extend(
            [ReflectionPad2d(3), nn.Conv2d(channels, config.output_nc, 7), nn.Tanh()]
        )
        self.upsampling = nn.Sequential(*layers)

    def forward(self, x, codes, control="full", query_points=None):
        traces = {}
        for index, block in enumerate(self.res_blocks):
            x, trace = block(x, codes, control, query_points)
            if trace is not None:
                traces[f"R{index + 1}"] = trace
        out = self.upsampling(x)
        return (out, traces) if query_points is not None else out


Decoder = AttResDecoder


class Generator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        channels = cfg.base_channels
        layers = [
            ReflectionPad2d(3),
            nn.Conv2d(cfg.input_nc, channels, 7),
            instance_norm(channels, cfg.stable_instance_norm),
            nn.ReLU(inplace=True),
        ]
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 3, 2, 1),
                    instance_norm(channels * 2, cfg.stable_instance_norm),
                    nn.ReLU(inplace=True),
                ]
            )
            channels *= 2
        self.encoder = nn.Sequential(*layers)
        self.decoder = AttResDecoder(cfg)

    def forward(self, image, codes, control=None, query_points=None):
        if image.ndim != 4 or image.shape[0] < 1 or not image.is_floating_point():
            raise ValueError("Generator input must be a nonempty floating BCHW tensor")
        if image.shape[1:] != (
            self.config.input_nc,
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError(
                "Input channels or image size disagree with model configuration"
            )
        control = self.config.attention_control if control is None else control
        return self.decoder(self.encoder(image), codes, control, query_points)


class Discriminator(nn.Module):
    def __init__(self, channels, base_channels=64, stable_instance_norm=False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        current = base_channels
        for stride in (2, 2, 1):
            layers.extend(
                [
                    nn.Conv2d(current, current * 2, 4, stride, 1),
                    instance_norm(current * 2, stable_instance_norm),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current *= 2
        layers.append(nn.Conv2d(current, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, image):
        return self.model(image)


def initialize_weights(module, scheme="kaiming"):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if scheme == "kaiming":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif scheme == "normal":
            nn.init.normal_(module.weight, 0, 0.02)
        else:
            raise ValueError("Unknown initialization")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
