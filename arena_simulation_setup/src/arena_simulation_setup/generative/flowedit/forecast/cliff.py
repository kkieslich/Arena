"""Online CLiFF-maps (Zhu et al., "Fast Online Learning of CLiFF-maps in Changing Environments", ICRA 2025, arXiv
2410.12237), projected onto the flow grid.

A CLiFF-map (circular-linear flow field, Kucner et al. 2017) keeps at every location a semi-wrapped Gaussian mixture
(SWGMM) over the velocity V = (theta, rho) seen there: direction theta in [0, 2 pi), wrapped, and speed rho, linear. Each
component j has a weight m_j, a mean mu_j and a 2 x 2 covariance Sigma_j; its density sums the normal over the winding
numbers w in {-1, 0, 1} (the paper's approximation of w in Z).

**Online update** (the paper's Algorithm 1), one iteration per batch of observations:
- a location with no model yet builds one from the batch: mean shift for the number and start of the modes, then EM;
- a location with a model updates it by stochastic EM: the sufficient statistics s = (eta, eta y, eta y y^T), averaged
  over the batch's N_k observations, move by s_k = s_{k-1} + gamma_k (S_k - s_{k-1}), and the M-step reads m, mu and
  Sigma off them. The step is gamma_k = N_k / N_ind^k with the decayed count N_ind^k = lambda N_ind^{k-1} + N_k, so
  older observations weigh geometrically less (decay rate lambda);
- a batch the model does not explain adds components: when its observations' mean mixture density falls under
  eta_thres = 0.1, a mixture is built from the batch alone and joins the old one, the new components weighted gamma_k
  and the old 1 - gamma_k. (The paper writes the test as the summed responsibilities under eta_thres; responsibilities
  sum to one per observation, so the test here reads the unnormalised responsibilities, the mixture density.)

Responsibilities run over components and winding numbers, so a mean near the wrap point takes the observations on
either side of it. The paper's convergence precision (1e-5) and iteration cap (100) hold for mean shift and EM.

**On this grid.** The paper's maps have 1 m locations and 1 Hz observations. Here the locations are the flow grid's
0.5 m cells and a track gives 2 observations per second, the same observations per location side. A location takes the
observations within `radius_m` of its centre. The paper leaves the radius, the mean-shift bandwidth and lambda open
(it reports lambda 0.1 to 0.9 as robust); here they are `CliffParams`, and `forecast baseline cliff` fits lambda on
the train partition's tracks-seen curve.

**Projection.** The SWGMM models motion: it is fitted to the observations at or above arena_evaluation's standing
speed (0.2 m/s), and a location's motion ratio p (the moving share of its decayed observation count, the CLiFF-map's
motion ratio) splits its mass. A cell's map is its decayed observation count, of which 1 - p stands and p is spread
over the 9 bins by integrating the SWGMM: heading bin b takes theta in its 45 degree sector at speed >= 0.2 m/s, the
standing bin takes speed < 0.2 m/s (the mixture's tail below the threshold). The integral over theta is a midpoint
rule on 32 points per sector, with the speed's conditional normal integrated exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import attrs

from .maps import BINS, HEADINGS, STANDING, FlowGrid

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

TWO_PI = 2.0 * math.pi
#: winding numbers the semi-wrapped normal sums over
WINDINGS = (-1, 0, 1)
#: arena_evaluation's STANDING_MPS: slower is standing
STANDING_MPS = 0.2
#: quadrature points per heading sector of the projection
_QUAD = 32
#: points at most one location's mean shift runs on (a seeded subsample beyond)
_SHIFT_POINTS = 256
#: a map entry below this share of the largest one is dropped
_NEGLIGIBLE = 1e-12
#: entries of the largest array one batch of EM or mean shift holds
_CHUNK = 6_000_000


@attrs.frozen
class CliffParams:
    """The online CLiFF-map's settings; the paper's where it names them."""

    #: observations per second of a track (the paper: 1 Hz on 1 m locations; here 0.5 m cells)
    rate_hz: float = 2.0
    #: a location takes the observations within this distance of its centre (m)
    radius_m: float = 0.5
    #: lambda: how much an iteration keeps of the decayed observation count
    decay: float = 0.9
    #: eta_thres: a batch whose mean mixture density is lower adds components
    novelty: float = 0.1
    #: mean shift's Gaussian kernel: direction (rad) and speed (m/s)
    bandwidth: tuple[float, float] = (0.5, 0.3)
    #: modes closer than this many bandwidths merge
    merge: float = 0.5
    #: components a location keeps at most (the smallest weights go)
    max_components: int = 6
    #: convergence precision and iteration cap of mean shift and EM (the paper's)
    tol: float = 1e-5
    max_iter: int = 100
    #: floor of the direction (rad) and speed (m/s) standard deviations, so a mode of a few equal velocities stays a density
    min_std: tuple[float, float] = (0.05, 0.02)

    def dump(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in attrs.asdict(self).items()}

    @classmethod
    def load(cls, doc: dict[str, Any]) -> CliffParams:
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in doc.items() if k in attrs.fields_dict(cls)})


# --- observations ----------------------------------------------------------------------------------------------------


@attrs.frozen
class Observations:
    """Velocity observations: position (m, world frame), direction theta in [0, 2 pi) and speed rho (m/s)."""

    x: Any
    y: Any
    theta: Any
    rho: Any

    def __len__(self) -> int:
        return len(self.x)

    @classmethod
    def concat(cls, parts: Iterable[Observations]) -> Observations:
        import numpy as np

        parts = list(parts)
        if not parts:
            return cls(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("x", "y", "theta", "rho")))


def observations(track: pl.DataFrame, rate_hz: float) -> Observations:
    """One pedestrian's track (the track store's t, x, y, vx, vy) as observations at `rate_hz`: the first frame of each
    1 / rate_hz interval, its velocity the recorded one, else the step to the next frame."""
    import numpy as np

    t = track["t"].to_numpy()
    order = np.argsort(t, kind="stable")
    t = t[order]
    x, y = track["x"].to_numpy()[order], track["y"].to_numpy()[order]
    vx, vy = track["vx"].to_numpy()[order].astype(float), track["vy"].to_numpy()[order].astype(float)
    if len(t) == 0:
        return Observations.concat([])
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        dvx = np.append(np.diff(x) / dt, np.nan)
        dvy = np.append(np.diff(y) / dt, np.nan)
    vx = np.where(np.isfinite(vx), vx, dvx)
    vy = np.where(np.isfinite(vy), vy, dvy)
    slot = np.floor((t - t[0]) * rate_hz).astype(np.int64)
    first = np.r_[True, slot[1:] != slot[:-1]]
    keep = first & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(x) & np.isfinite(y)
    vx, vy = vx[keep], vy[keep]
    return Observations(x[keep], y[keep], np.mod(np.arctan2(vy, vx), TWO_PI), np.hypot(vx, vy))


def track_observations(tracks: pl.DataFrame, rate_hz: float) -> dict[str, Observations]:
    """Each pedestrian's observations by track key."""
    return {str(key): observations(part, rate_hz) for (key,), part in tracks.group_by(["key"], maintain_order=True)}


# --- the semi-wrapped normal -----------------------------------------------------------------------------------------


def _gauss(d0: np.ndarray, d1: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Bivariate normal density of offsets (d0, d1) under covariances `cov` (..., 2, 2), broadcast."""
    import numpy as np

    a, b, c = cov[..., 0, 0], cov[..., 1, 1], cov[..., 0, 1]
    det = np.maximum(a * b - c * c, 1e-18)
    quad = (b * d0 * d0 - 2 * c * d0 * d1 + a * d1 * d1) / det
    return np.exp(-0.5 * np.minimum(quad, 700.0)) / (TWO_PI * np.sqrt(det))


def _floor_cov(cov: np.ndarray, min_std: tuple[float, float]) -> np.ndarray:
    """Covariances with their variances floored and their correlation kept within 0.99."""
    import numpy as np

    cov = cov.copy()
    cov[..., 0, 0] = np.maximum(cov[..., 0, 0], min_std[0] ** 2)
    cov[..., 1, 1] = np.maximum(cov[..., 1, 1], min_std[1] ** 2)
    lim = 0.99 * np.sqrt(cov[..., 0, 0] * cov[..., 1, 1])
    c = np.clip(cov[..., 0, 1], -lim, lim)
    cov[..., 0, 1] = c
    cov[..., 1, 0] = c
    return cov


def _joint(theta: np.ndarray, rho: np.ndarray, m: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(B, n, J, W): m_j times the normal of each observation at each winding number; points (B, n), components (B, J)."""
    import numpy as np

    w = np.asarray(WINDINGS, float) * TWO_PI
    d0 = theta[:, :, None, None] + w[None, None, None, :] - mu[:, None, :, None, 0]
    d1 = (rho[:, :, None] - mu[:, None, :, 1])[..., None]
    return m[:, None, :, None] * _gauss(d0, d1, cov[:, None, :, None])


def _em(theta: np.ndarray, rho: np.ndarray, weight: np.ndarray, m: np.ndarray, mu: np.ndarray, cov: np.ndarray, params: CliffParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EM of the SWGMMs of B locations: points (B, n) with weights (0 pads), start components (B, J) (weight 0 unused)."""
    import numpy as np

    w = np.asarray(WINDINGS, float) * TWO_PI
    total = np.maximum(weight.sum(axis=1), 1e-12)
    last = np.full(len(theta), -np.inf)
    active = np.ones(len(theta), bool)
    for _ in range(params.max_iter):
        joint = _joint(theta[active], rho[active], m[active], mu[active], cov[active])
        dens = joint.sum(axis=(2, 3))
        r = joint / np.maximum(dens, 1e-300)[:, :, None, None] * weight[active][:, :, None, None]
        ll = (weight[active] * np.log(np.maximum(dens, 1e-300))).sum(axis=1) / total[active]
        nj = r.sum(axis=(1, 3))
        ut = theta[active][:, :, None, None] + w[None, None, None, :]
        ur = np.broadcast_to(rho[active][:, :, None, None], ut.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            mt = (r * ut).sum(axis=(1, 3)) / nj
            mr = (r * ur).sum(axis=(1, 3)) / nj
            d0, d1 = ut - mt[:, None, :, None], ur - mr[:, None, :, None]
            c00 = (r * d0 * d0).sum(axis=(1, 3)) / nj
            c11 = (r * d1 * d1).sum(axis=(1, 3)) / nj
            c01 = (r * d0 * d1).sum(axis=(1, 3)) / nj
        live = nj > 1e-9
        new_mu = np.stack([np.mod(mt, TWO_PI), mr], axis=-1)
        new_cov = np.stack([np.stack([c00, c01], -1), np.stack([c01, c11], -1)], -2)
        idx = np.flatnonzero(active)
        m[idx] = np.where(live, nj / total[active][:, None], 0.0)
        mu[idx] = np.where(live[..., None], new_mu, mu[idx])
        cov[idx] = _floor_cov(np.where(live[..., None, None], np.nan_to_num(new_cov), cov[idx]), params.min_std)
        done = np.abs(ll - last[active]) < params.tol
        last[idx] = ll
        active[idx[done]] = False
        if not active.any():
            break
    return m, mu, cov


def _wrap(d: np.ndarray) -> np.ndarray:
    import numpy as np

    return d - TWO_PI * np.rint(d / TWO_PI)


def _mean_shift(theta: np.ndarray, rho: np.ndarray, weight: np.ndarray, params: CliffParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean shift on B locations' points (B, n) on the cylinder; per location up to max_components modes, as start
    components (B, J): weight the share of points a mode takes, mean the mode, covariance that of its points."""
    import numpy as np

    ht, hr = params.bandwidth
    B, n = theta.shape
    xt, xr = theta.copy(), rho.copy()
    moving = weight > 0
    t32, r32, w32 = theta.astype(np.float32), rho.astype(np.float32), weight.astype(np.float32)
    active = np.flatnonzero(moving.any(axis=1))
    for _ in range(params.max_iter):
        if not len(active):
            break
        at, ar = xt[active].astype(np.float32), xr[active].astype(np.float32)
        dt = t32[active][:, None, :] - at[:, :, None]
        dt -= np.float32(TWO_PI) * np.rint(dt / np.float32(TWO_PI))
        dr = r32[active][:, None, :] - ar[:, :, None]
        k = np.exp(-0.5 * ((dt / ht) ** 2 + (dr / hr) ** 2)) * w32[active][:, None, :]
        ks = np.maximum(k.sum(axis=2), 1e-30)
        st = (k * dt).sum(axis=2) / ks
        nr = (k * r32[active][:, None, :]).sum(axis=2) / ks
        step = np.hypot(st / ht, (nr - ar) / hr)
        xt[active] = np.mod(at + st, TWO_PI)
        xr[active] = nr
        active = active[np.max(np.where(moving[active], step, 0.0), axis=1) >= params.tol]
    import scipy.sparse
    import scipy.sparse.csgraph

    # modes: converged points closer than `merge` bandwidths are one (connected components of that graph per location)
    near = np.hypot(_wrap(xt[:, :, None] - xt[:, None, :]) / ht, (xr[:, :, None] - xr[:, None, :]) / hr) < params.merge
    near &= moving[:, :, None] & moving[:, None, :]
    b_, i_, j_ = np.nonzero(near)
    size = B * n
    graph = scipy.sparse.coo_matrix((np.ones(len(b_)), (b_ * n + i_, b_ * n + j_)), shape=(size, size))
    _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    flat_w = weight.ravel()
    clusters = int(label.max()) + 1
    cw = np.bincount(label, weights=flat_w, minlength=clusters)
    # a cluster's mode: its first point's converged position; its covariance: that of its points about the mode
    first = np.full(clusters, -1)
    first[label[::-1]] = np.arange(size)[::-1]
    mode_t, mode_r = xt.ravel()[first], xr.ravel()[first]
    d0 = _wrap(theta.ravel() - mode_t[label])
    d1 = rho.ravel() - mode_r[label]
    with np.errstate(invalid="ignore", divide="ignore"):
        c00 = np.bincount(label, weights=flat_w * d0 * d0, minlength=clusters) / cw
        c11 = np.bincount(label, weights=flat_w * d1 * d1, minlength=clusters) / cw
        c01 = np.bincount(label, weights=flat_w * d0 * d1, minlength=clusters) / cw
    owner = first // n
    keep = cw > 0
    # per location, its heaviest max_components modes
    J = params.max_components
    order = np.lexsort((-cw, owner))
    order = order[keep[order]]
    rank = np.zeros(len(order), np.int64)
    if len(order):
        starts = np.r_[True, owner[order][1:] != owner[order][:-1]]
        group_start = np.maximum.accumulate(np.where(starts, np.arange(len(order)), 0))
        rank = np.arange(len(order)) - group_start
    sel = order[rank < J]
    slot = rank[rank < J]
    m = np.zeros((B, J))
    mu = np.zeros((B, J, 2))
    cov = np.zeros((B, J, 2, 2))
    ob = owner[sel]
    m[ob, slot] = cw[sel]
    mu[ob, slot, 0], mu[ob, slot, 1] = mode_t[sel], mode_r[sel]
    cov[ob, slot, 0, 0], cov[ob, slot, 1, 1] = np.nan_to_num(c00[sel]), np.nan_to_num(c11[sel])
    cov[ob, slot, 0, 1] = cov[ob, slot, 1, 0] = np.nan_to_num(c01[sel])
    m /= np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
    return m, mu, _floor_cov(cov, params.min_std)


def _padded(groups: Sequence[np.ndarray], theta: np.ndarray, rho: np.ndarray, cap: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points of each group (index arrays) padded to one width, at most `cap` per group (a seeded subsample)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    picked = [g if cap is None or len(g) <= cap else np.sort(rng.choice(g, cap, replace=False)) for g in groups]
    width = max((len(g) for g in picked), default=0)
    t = np.zeros((len(picked), width))
    r = np.zeros((len(picked), width))
    w = np.zeros((len(picked), width))
    for b, g in enumerate(picked):
        t[b, : len(g)], r[b, : len(g)], w[b, : len(g)] = theta[g], rho[g], 1.0
    return t, r, w


def _chunks(groups: Sequence[np.ndarray], per_entry: int) -> list[list[int]]:
    """Group indices in batches whose padded arrays stay under _CHUNK entries, similar sizes together."""
    import numpy as np

    order = np.argsort([len(g) for g in groups], kind="stable")
    out: list[list[int]] = []
    batch: list[int] = []
    for i in order:
        width = len(groups[i])
        if batch and (len(batch) + 1) * width * per_entry > _CHUNK:
            out.append(batch)
            batch = []
        batch.append(int(i))
    if batch:
        out.append(batch)
    return out


def build(groups: Sequence[np.ndarray], theta: np.ndarray, rho: np.ndarray, params: CliffParams, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The SWGMM of each group of observations (index arrays into theta, rho): mean shift, then EM on all its points."""
    import numpy as np

    J = params.max_components
    m = np.zeros((len(groups), J))
    mu = np.zeros((len(groups), J, 2))
    cov = np.zeros((len(groups), J, 2, 2))
    for batch in _chunks([g[:_SHIFT_POINTS] for g in groups], min(_SHIFT_POINTS, max((len(g) for g in groups), default=1))):
        t, r, w = _padded([groups[i] for i in batch], theta, rho, _SHIFT_POINTS, seed)
        m[batch], mu[batch], cov[batch] = _mean_shift(t, r, w, params)
    for batch in _chunks(groups, J * len(WINDINGS) * 4):
        t, r, w = _padded([groups[i] for i in batch], theta, rho, None, seed)
        m[batch], mu[batch], cov[batch] = _em(t, r, w, m[batch], mu[batch], cov[batch], params)
    return m, mu, cov


# --- the map ---------------------------------------------------------------------------------------------------------


@attrs.define
class CliffMap:
    """An online CLiFF-map on the flow grid's cells: per location the decayed observation counts (all, moving, and the
    sEM's N_ind) and an SWGMM of up to `max_components` components (weight 0 unused)."""

    grid: FlowGrid
    params: CliffParams
    n_all: Any
    n_moving: Any
    n_ind: Any
    m: Any
    mu: Any
    cov: Any
    built: Any
    iterations: int = 0
    #: each location's bin shares as last projected, and the locations whose model changed since
    _shares: Any = None
    _dirty: Any = None

    @classmethod
    def empty(cls, grid: FlowGrid, params: CliffParams = CliffParams()) -> CliffMap:
        import numpy as np

        L, J = grid.ny * grid.nx, params.max_components
        return cls(grid, params, np.zeros(L), np.zeros(L), np.zeros(L), np.zeros((L, J)), np.zeros((L, J, 2)), np.zeros((L, J, 2, 2)), np.zeros(L, bool))

    def copy(self) -> CliffMap:
        """A copy of the counts, the mixtures and the projection cache; the grid and params are shared."""

        def arr(a: Any) -> Any:  # noqa: ANN401
            return None if a is None else a.copy()

        return CliffMap(self.grid, self.params, *(arr(getattr(self, f)) for f in ("n_all", "n_moving", "n_ind", "m", "mu", "cov", "built")), self.iterations, arr(self._shares), arr(self._dirty))

    def locations(self, obs: Observations) -> tuple[np.ndarray, np.ndarray]:
        """(observation, location) pairs: each observation and every room cell whose centre lies within radius_m of it."""
        import numpy as np

        g = self.grid
        rooms = g.room_cells().ravel()
        reach = int(math.ceil(self.params.radius_m / g.cell))
        i0, j0, _ = g.index(obs.x, obs.y)
        pairs_o, pairs_l = [], []
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                i, j = i0 + di, j0 + dj
                ok = (i >= 0) & (i < g.ny) & (j >= 0) & (j < g.nx)
                cx, cy = g.x0 + (j + 0.5) * g.cell, g.y0 + (i + 0.5) * g.cell
                ok &= np.hypot(obs.x - cx, obs.y - cy) <= self.params.radius_m + 1e-9
                loc = np.where(ok, i * g.nx + j, 0)
                ok &= rooms[loc]
                pairs_o.append(np.flatnonzero(ok))
                pairs_l.append(loc[ok])
        return np.concatenate(pairs_o), np.concatenate(pairs_l)

    def update(self, obs: Observations, seed: int = 0) -> CliffMap:
        """One iteration of Algorithm 1 with the batch `obs`, in place: decay, then build or sEM-update every location it reaches."""
        import numpy as np

        p = self.params
        lam = p.decay
        self.n_all *= lam
        self.n_moving *= lam
        self.n_ind *= lam
        self.iterations += 1
        if len(obs) == 0:
            return self
        o, loc = self.locations(obs)
        L = len(self.n_all)
        self.n_all += np.bincount(loc, minlength=L)
        moving = obs.rho[o] >= STANDING_MPS
        o, loc = o[moving], loc[moving]
        if len(o) == 0:
            return self
        n_k = np.bincount(loc, minlength=L).astype(float)
        self.n_moving += n_k
        self.n_ind += n_k
        order = np.argsort(loc, kind="stable")
        o, loc = o[order], loc[order]
        reached, start = np.unique(loc, return_index=True)
        if self._dirty is not None:
            self._dirty[reached] = True
        groups = np.split(o, start[1:])
        gamma = n_k[reached] / np.maximum(self.n_ind[reached], 1e-12)
        fresh = ~self.built[reached]
        if fresh.any():
            idx = np.flatnonzero(fresh)
            m, mu, cov = build([groups[i] for i in idx], obs.theta, obs.rho, p, seed)
            self.m[reached[idx]], self.mu[reached[idx]], self.cov[reached[idx]] = m, mu, cov
            self.built[reached[idx]] = True
        if (~fresh).any():
            self._sem([groups[i] for i in np.flatnonzero(~fresh)], reached[~fresh], gamma[~fresh], obs, seed)
        return self

    def _sem(self, groups: Sequence[np.ndarray], locs: np.ndarray, gamma: np.ndarray, obs: Observations, seed: int) -> None:
        """The stochastic EM step at `locs`, or new components where the batch is not explained."""
        import numpy as np

        p = self.params
        J = p.max_components
        w = np.asarray(WINDINGS, float) * TWO_PI
        for batch in _chunks(groups, J * len(WINDINGS) * 4):
            loc, g = locs[batch], gamma[batch]
            t, r, wt = _padded([groups[i] for i in batch], obs.theta, obs.rho, None, seed)
            m, mu, cov = self.m[loc], self.mu[loc], self.cov[loc]
            joint = _joint(t, r, m, mu, cov)
            dens = joint.sum(axis=(2, 3))
            n = np.maximum(wt.sum(axis=1), 1e-12)
            novel = (wt * dens).sum(axis=1) / n < p.novelty
            # sufficient statistics of the batch, per observation (S_k), over components and winding numbers
            resp = joint / np.maximum(dens, 1e-300)[:, :, None, None] * wt[:, :, None, None]
            ut = t[:, :, None, None] + w[None, None, None, :]
            ur = np.broadcast_to(r[:, :, None, None], ut.shape)
            s1 = resp.sum(axis=(1, 3)) / n[:, None]
            s2 = np.stack([(resp * ut).sum(axis=(1, 3)), (resp * ur).sum(axis=(1, 3))], -1) / n[:, None, None]
            s3 = np.stack(
                [np.stack([(resp * ut * ut).sum(axis=(1, 3)), (resp * ut * ur).sum(axis=(1, 3))], -1), np.stack([(resp * ut * ur).sum(axis=(1, 3)), (resp * ur * ur).sum(axis=(1, 3))], -1)], -2
            ) / n[:, None, None, None]
            # the model's statistics: what its M-step reads back
            h1 = m
            h2 = m[..., None] * mu
            h3 = m[..., None, None] * (cov + mu[..., :, None] * mu[..., None, :])
            gg = g[:, None]
            n1 = h1 + gg * (s1 - h1)
            n2 = h2 + gg[..., None] * (s2 - h2)
            n3 = h3 + gg[..., None, None] * (s3 - h3)
            live = n1 > 1e-12
            with np.errstate(invalid="ignore", divide="ignore"):
                new_mu = n2 / n1[..., None]
                new_cov = n3 / n1[..., None, None] - new_mu[..., :, None] * new_mu[..., None, :]
            new_mu = np.where(live[..., None], np.stack([np.mod(new_mu[..., 0], TWO_PI), new_mu[..., 1]], -1), mu)
            new_cov = _floor_cov(np.where(live[..., None, None], np.nan_to_num(new_cov), cov), p.min_std)
            new_m = np.where(live, n1, 0.0)
            new_m /= np.maximum(new_m.sum(axis=1, keepdims=True), 1e-12)
            keep = ~novel
            self.m[loc[keep]], self.mu[loc[keep]], self.cov[loc[keep]] = new_m[keep], new_mu[keep], new_cov[keep]
            if novel.any():
                idx = np.flatnonzero(novel)
                bm, bmu, bcov = build([groups[batch[i]] for i in idx], obs.theta, obs.rho, p, seed)
                for q, i in enumerate(idx):
                    self._join(int(loc[i]), float(g[i]), bm[q], bmu[q], bcov[q])

    def _join(self, loc: int, gamma: float, m: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> None:
        """Old components weighted 1 - gamma and new ones gamma, normalised, the heaviest max_components kept."""
        import numpy as np

        allm = np.concatenate([self.m[loc] * (1 - gamma), m * gamma])
        allmu = np.concatenate([self.mu[loc], mu])
        allcov = np.concatenate([self.cov[loc], cov])
        top = np.argsort(-allm, kind="stable")[: self.params.max_components]
        self.m[loc] = allm[top] / max(float(allm[top].sum()), 1e-12)
        self.mu[loc], self.cov[loc] = allmu[top], allcov[top]

    def shares(self) -> np.ndarray:
        """(L, BINS): each built location's SWGMM integrated over the 8 heading sectors at speed >= STANDING_MPS and over
        speed < STANDING_MPS (the standing bin), rows summing to one."""
        import numpy as np
        import scipy.special

        L, J = self.m.shape
        if self._shares is None:
            self._shares, self._dirty = np.zeros((L, BINS)), np.ones(L, bool)
        out = self._shares
        idx = np.flatnonzero(self.built & self._dirty)
        self._dirty[:] = False
        if not len(idx):
            return out.copy()
        m, mu, cov = self.m[idx], self.mu[idx], self.cov[idx]
        q = _QUAD * HEADINGS
        theta = -math.pi + (np.arange(q) + 0.5) * TWO_PI / q
        step = TWO_PI / q
        s0 = np.sqrt(cov[..., 0, 0])
        s1 = np.sqrt(cov[..., 1, 1])
        beta = cov[..., 0, 1] / cov[..., 0, 0]
        cond = np.sqrt(np.maximum(cov[..., 1, 1] - beta * cov[..., 0, 1], 1e-12))
        moving = np.zeros((len(idx), q))
        for k in (-2, -1, 0, 1, 2):
            d = theta[None, None, :] + k * TWO_PI - mu[..., None, 0]
            f = np.exp(-0.5 * (d / s0[..., None]) ** 2) / (math.sqrt(TWO_PI) * s0[..., None])
            above = scipy.special.ndtr((mu[..., None, 1] + beta[..., None] * d - STANDING_MPS) / cond[..., None])
            moving += (m[..., None] * f * above).sum(axis=1) * step
        sector = moving.reshape(len(idx), HEADINGS, _QUAD).sum(axis=2)
        standing = (m * scipy.special.ndtr((STANDING_MPS - mu[..., 1]) / s1)).sum(axis=1)
        out[idx, :HEADINGS] = sector
        out[idx, STANDING] = standing
        out[idx] /= np.maximum(out[idx].sum(axis=1, keepdims=True), 1e-12)
        return out.copy()

    def map(self) -> np.ndarray:
        """The map (ny, nx, BINS): each cell's decayed observation count, its standing share (1 - motion ratio) in the
        standing bin and its motion ratio spread over the bins by the SWGMM."""
        import numpy as np

        g = self.grid
        out = np.zeros((len(self.n_all), BINS))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(self.n_all > 0, np.minimum(self.n_moving / self.n_all, 1.0), 0.0)
        out += self.shares() * (self.n_all * ratio)[:, None]
        out[:, STANDING] += self.n_all * (1.0 - ratio)
        # a mixture's far tail integrates to denormal numbers, whose halves underflow to 0 inside a JSD; they hold no mass
        out[out < _NEGLIGIBLE * max(float(out.max()), 1e-300)] = 0.0
        return out.reshape(g.ny, g.nx, BINS)


def from_tracks(grid: FlowGrid, tracks: Iterable[pl.DataFrame], params: CliffParams = CliffParams(), seed: int = 0) -> CliffMap:
    """A CLiFF-map built in one iteration from every observation of `tracks` (the history model of a pre-edit map)."""
    obs = Observations.concat(o for t in tracks for o in track_observations(t, params.rate_hz).values())
    return CliffMap.empty(grid, params).update(obs, seed)


__all__ = ["STANDING_MPS", "CliffMap", "CliffParams", "Observations", "build", "from_tracks", "observations", "track_observations"]
