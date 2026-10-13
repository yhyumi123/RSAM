"""Patch contrastive supervision with registered projection heads."""
import torch
from torch import nn
from torch.nn import functional as F

class ProjectionHeads(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(c, 256), nn.ReLU(), nn.Linear(256, 256))
                                    for c in (base_channels, 2 * base_channels, 4 * base_channels)])


def features(generator, image):
    values = []
    x = image
    for index, layer in enumerate(generator.encoder):
        x = layer(x)
        if index in (3, 6):
            values.append(x.clone())
    for layer in generator.decoder.res_blocks[0].conv_block:
        x = layer(x)
    values.append(x.clone())
    return values


def patch_nce(generator, heads, source, generated, step, seed, branch):
    with torch.no_grad():
        keys = features(generator, source)
    queries = features(generator, generated)
    rng = torch.Generator(device='cpu').manual_seed(seed + step * 104729 + branch * 10000019)
    per_image, counts = [], []
    for row in range(len(source)):
        terms = []
        row_counts = []
        for head, key, query in zip(heads.heads, keys, queries):
            # Prefer foreground locations for feature sampling.
            visible = F.interpolate((source[row:row+1, source.shape[1] // 2:source.shape[1] // 2 + 1].detach() > -.99).float(),
                                    size=key.shape[-2:], mode='nearest')[0, 0].bool().cpu()
            candidates = visible.flatten().nonzero(as_tuple=True)[0]
            # Sample the full plane when foreground support is empty.
            if len(candidates) < 16:
                candidates = torch.arange(key.shape[-2] * key.shape[-1])
            ids = candidates[torch.randperm(len(candidates), generator=rng)[:256]].to(source.device)
            k = key[row].flatten(1).T.index_select(0, ids)
            q = query[row].flatten(1).T.index_select(0, ids)
            with torch.no_grad():
                k = F.normalize(head(k), dim=1, eps=1e-12)
            q = F.normalize(head(q), dim=1, eps=1e-12)
            terms.append(F.cross_entropy(q @ k.T / .07, torch.arange(len(ids), device=q.device)))
            row_counts.append(len(ids))
        per_image.append(torch.stack(terms).mean())
        counts.append(row_counts)
    return torch.stack(per_image).mean()

