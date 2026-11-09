"""A PNG per world: the recorded pooled occupancy, the forecast, the stale base map, and where the forecast is off."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .maps import STANDING, FlowGrid, normalized, occupancy

if TYPE_CHECKING:
    import numpy as np

    from .model import Scene


def plot_world(path: Path, grid: FlowGrid, truth: np.ndarray, forecast: np.ndarray, base_map: np.ndarray | None, scene: Scene, title: str = "") -> Path:
    """Write the panels of one world to `path`: shares of time per cell, each map normalized, the demand objects marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import PowerNorm, TwoSlopeNorm

    def share(m: np.ndarray) -> np.ndarray:
        n = normalized(occupancy(m))
        return np.zeros(m.shape[:2]) if n is None else n[:, :, 0]

    panels = [("recorded (pooled)", share(truth)), ("forecast", share(forecast))]
    if base_map is not None:
        panels.append(("stale (base world)", share(base_map)))
    diff = share(forecast) - share(truth)
    extent = (grid.x0, grid.x0 + grid.nx * grid.cell, grid.y0, grid.y0 + grid.ny * grid.cell)
    vmax = max(float(p.max()) for _, p in panels) or 1.0
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 4.2 * grid.ny / max(grid.nx, 1) + 0.9), squeeze=False)
    rooms = grid.room_cells()
    for ax, (name, m) in zip(axes[0], panels, strict=False):
        shown = np.ma.masked_where(~rooms, m)
        im = ax.imshow(shown, origin="lower", extent=extent, cmap="magma", norm=PowerNorm(0.5, vmin=0.0, vmax=vmax))
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes[0][: len(panels)].tolist(), shrink=0.8, label="share of pedestrian-seconds per cell")
    lim = float(np.abs(diff).max()) or 1e-9
    ax = axes[0][-1]
    im = ax.imshow(np.ma.masked_where(~rooms, diff), origin="lower", extent=extent, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim))
    ax.set_title("forecast - recorded", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes[0]:
        for d in scene.demands:
            ax.plot(*d.at, marker="s", color="cyan", markersize=5, markeredgecolor="black")
            ax.plot(*d.spot(0), marker="o", color="lime", markersize=3)
        ax.set_xticks([])
        ax.set_yticks([])
    stood = [float(m[:, :, STANDING].sum()) / max(float(m.sum()), 1e-9) for m in (truth, forecast)]
    fig.suptitle(f"{title}\nstanding share: recorded {stood[0]:.2f}, forecast {stood[1]:.2f}; squares: demand objects, dots: where they are used", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["plot_world"]
