"""The learned forecaster's network (torch): a small typed GNN over the graph, its embeddings written onto the grid, a
per-need encoder, and a U-Net to 9 non-negative rates per cell (8 headings and standing).

- **GNN**: each node kind (room, door, demand object, need) is embedded by an MLP of its features; `ROUNDS` rounds of
  typed message passing along the graph's edges (both directions, the walking distance as edge feature, sums), with a
  residual MLP update. It is how objects that compete for one need, their capacities and the rooms people start in meet.
- **onto the grid**: room embeddings on the room's cells, door embeddings on its span, demand embeddings on the object's
  front and queue spots; needs reach the grid through their objects and rooms.
- **need encoder**: the same two convolutions on each need's stack of channels, summed over needs, so any number of needs
  works.
- **U-Net**: the channels, the need features and the GNN layers, `LEVELS` times halved, out to 9 rates per cell scaled by
  the crowd's pedestrian-seconds per room cell and kept to the rooms. Fully convolutional, so any world size works.

The D4 symmetries (quarter turns, mirror) act on the grid input and output together, the heading bins permuted with them
(`transform`): training augments with them, prediction averages over all eight.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .maps import BINS, HEADINGS, STANDING
from .raster import CHANNELS, EDGE_KINDS, NEED_CHANNELS, NODE_FEATURES, NODE_KINDS, Raster

DIM = 16
ROUNDS = 3
NEED_DIM = 8
WIDTHS = (32, 64, 96, 128)
LEVELS = len(WIDTHS) - 1
#: node kinds whose embeddings are written onto the grid
WRITTEN = ("room", "door", "demand")


def _mlp(i: int, o: int, hidden: int = 32) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, o))


def heading_perm(flip: bool, k: int) -> list[int]:
    """new bin of each old bin under the mirror x -> -x (if `flip`) then `k` quarter turns counterclockwise; standing stays."""
    out = []
    for b in range(HEADINGS):
        n = (3 - b) % HEADINGS if flip else b
        out.append((n + 2 * k) % HEADINGS)
    return [*out, STANDING]


def transform(t: torch.Tensor, flip: bool, k: int, bins: bool = False) -> torch.Tensor:
    """`t` (..., ny, nx; row i at y0 + i * cell, column j at x0 + j * cell) mirrored x -> -x (if `flip`), then turned `k`
    quarter turns counterclockwise in the world frame; `bins`: dim -3 holds the BINS rates and is permuted with it."""
    if flip:
        t = t.flip(-1)
    for _ in range(k % 4):
        t = t.transpose(-1, -2).flip(-1)
    if bins:
        perm = heading_perm(flip, k)
        inverse = [0] * BINS
        for old, new in enumerate(perm):
            inverse[new] = old
        t = t[..., inverse, :, :]
    return t


def untransform(t: torch.Tensor, flip: bool, k: int, bins: bool = False) -> torch.Tensor:
    """The inverse of `transform`."""
    if bins:
        perm = heading_perm(flip, k)
        t = t[..., perm, :, :]
    for _ in range(k % 4):
        t = t.flip(-1).transpose(-1, -2)
    if flip:
        t = t.flip(-1)
    return t


class GNN(nn.Module):
    def __init__(self, dim: int = DIM, rounds: int = ROUNDS) -> None:
        super().__init__()
        self.dim, self.rounds = dim, rounds
        self.embed = nn.ModuleDict({k: _mlp(len(NODE_FEATURES[k]), dim) for k in NODE_KINDS})
        self.message = nn.ModuleDict({f"{name}_{way}": _mlp(dim + 1, dim) for name, _, _ in EDGE_KINDS for way in ("fwd", "bwd")})
        self.update = nn.ModuleDict({k: _mlp(2 * dim, dim) for k in NODE_KINDS})

    def forward(self, x: Mapping[str, torch.Tensor], edges: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        h = {k: self.embed[k](x[k]) if len(x[k]) else x[k].new_zeros((0, self.dim)) for k in NODE_KINDS}
        for _ in range(self.rounds):
            agg = {k: torch.zeros_like(h[k]) for k in NODE_KINDS}
            for name, a, b in EDGE_KINDS:
                src, dst, dist = edges[name]
                if len(src) == 0:
                    continue
                agg[b] = agg[b].index_add(0, dst, self.message[f"{name}_fwd"](torch.cat([h[a][src], dist], 1)))
                agg[a] = agg[a].index_add(0, src, self.message[f"{name}_bwd"](torch.cat([h[b][dst], dist], 1)))
            h = {k: h[k] + self.update[k](torch.cat([h[k], agg[k]], 1)) if len(h[k]) else h[k] for k in NODE_KINDS}
        return h


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int = BINS, widths: tuple[int, ...] = WIDTHS) -> None:
        super().__init__()
        self.down = nn.ModuleList([_block(cin, widths[0]), *(_block(widths[i], widths[i + 1]) for i in range(len(widths) - 1))])
        self.up = nn.ModuleList([_block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1))])
        self.out = nn.Conv2d(widths[0], cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.out(x)


class Forecaster(nn.Module):
    """Raster in, (BINS, ny, nx) expected pedestrian-seconds out."""

    def __init__(self, demand: bool = True) -> None:
        super().__init__()
        self.demand = demand
        self.gnn = GNN()
        self.need = nn.Sequential(nn.Conv2d(len(NEED_CHANNELS), 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, NEED_DIM, 3, padding=1))
        self.unet = UNet(len(CHANNELS) + NEED_DIM + DIM * len(WRITTEN))

    def inputs(self, r: Tensors) -> torch.Tensor:
        """(C, ny, nx): the raster's channels, the need features and the GNN embeddings on their cells, in the world's frame."""
        h = self.gnn(r.x, r.edges)
        layers = [r.channels]
        if r.need_channels.shape[0]:
            layers.append(self.need(r.need_channels).sum(0))
        else:
            layers.append(r.channels.new_zeros((NEED_DIM, *r.channels.shape[-2:])))
        ny, nx = r.channels.shape[-2:]
        for k in WRITTEN:
            node, cell = r.cells[k]
            flat = r.channels.new_zeros((ny * nx, DIM))
            if len(node):
                flat = flat.index_add(0, cell, h[k][node])
            layers.append(flat.T.reshape(DIM, ny, nx))
        return torch.cat(layers, 0)

    def forward(self, batch: list[tuple[Tensors, bool, int]]) -> list[torch.Tensor]:
        """Rates of each (raster, flip, k) in its transformed frame, (BINS, ny', nx') each."""
        xs, masks, scales = [], [], []
        for r, flip, k in batch:
            xs.append(transform(self.inputs(r), flip, k))
            masks.append(transform(r.mask, flip, k))
            scales.append(r.scale)
        hmax = max(x.shape[-2] for x in xs)
        wmax = max(x.shape[-1] for x in xs)
        step = 2**LEVELS
        hp, wp = -(-hmax // step) * step, -(-wmax // step) * step
        padded = torch.stack([F.pad(x, (0, wp - x.shape[-1], 0, hp - x.shape[-2])) for x in xs])
        z = self.unet(padded)
        out = []
        for i, x in enumerate(xs):
            h, w = x.shape[-2:]
            out.append(F.softplus(z[i, :, :h, :w]) * scales[i] * masks[i])
        return out


class Tensors:
    """A Raster as tensors on one device."""

    def __init__(self, r: Raster, device: torch.device | str, demand: bool = True) -> None:
        import numpy as np

        t = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        self.channels = t(r.channels)
        self.need_channels = t(r.need_channels) if demand else t(np.zeros((0, *r.need_channels.shape[1:]), np.float32))
        kinds = NODE_KINDS if demand else ("room", "door")
        self.x = {k: t(r.nodes[k].x) if k in kinds else t(np.zeros((0, len(NODE_FEATURES[k])), np.float32)) for k in NODE_KINDS}
        self.edges = {}
        for name, a, b in EDGE_KINDS:
            e = r.edges[name]
            keep = a in kinds and b in kinds
            self.edges[name] = (t(e.src if keep else [], torch.int64), t(e.dst if keep else [], torch.int64), t(e.dist if keep else np.zeros((0, 1)), torch.float32).reshape(-1, 1))
        self.cells = {}
        for k in WRITTEN:
            nodes = r.nodes[k] if k in kinds else None
            pairs = [(i, int(c)) for i, cells in enumerate(nodes.cells if nodes else ()) for c in cells]
            self.cells[k] = (t([p[0] for p in pairs], torch.int64), t([p[1] for p in pairs], torch.int64))
        self.mask = t(r.mask.astype(np.float32))
        self.scale = float(r.scale)


def poisson_nll(rate: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor, coarse: float = 0.5) -> torch.Tensor:
    """Poisson NLL of `counts` (BINS, ny, nx) under `rate` on the mask's cells, per pedestrian-second recorded, plus `coarse`
    times the same on 2 x 2 blocks (1 m cells)."""
    eps = 1e-6
    total = counts.sum().clamp_min(1.0)
    fine = ((rate - counts * torch.log(rate + eps)) * mask).sum() / total
    if not coarse:
        return fine
    r2, c2, m2 = (F.avg_pool2d(a[None], 2, ceil_mode=True)[0] * 4 for a in (rate, counts, mask[None].expand_as(rate)))
    m2 = (m2 > 0).float()
    return fine + coarse * ((r2 - c2 * torch.log(r2 + eps)) * m2).sum() / total


__all__ = ["DIM", "GNN", "Forecaster", "Tensors", "UNet", "heading_perm", "poisson_nll", "transform", "untransform"]
