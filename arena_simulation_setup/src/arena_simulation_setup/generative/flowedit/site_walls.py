"""Walls a site's world holds for objects that are not part of the site: things the scan caught that do not normally
stand there (a temporary sign, say). A site config lists them (`removed_walls`) by level, wall kind and bounding box.
`flowedit infra remove-walls` takes them out of the site's world and of every layout copy of it in place: the wall
loop from `world.yaml` (the runtime renders the occupancy map, the simulators' walls and HumanSim's from it) and its
pixels from the preview map `map/<level>.png` (read with `debug.map_source:=disk`). Every other byte of `world.yaml`
and every other pixel stays as it is. The INFRA writers refuse a site world that still holds such a wall
(`check_site`), so layout copies saved from it are free of it too.

A wall is matched by its geometry, not by its index: a closed path of the listed kind on the listed level whose
bounding box lies within `MATCH_M` of the listed one, side by side."""

from __future__ import annotations

import math
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import attrs

#: how far each side of a wall's bounding box may lie from the listed one for the wall to match
MATCH_M = 0.05
#: a map pixel whose centre lies within this many pixels of a removed wall's path is that wall's
PIXEL_REACH = 1.5


@attrs.frozen
class RemovedWall:
    """A wall loop the site's world holds for an object that is not part of the site: `name`, why (`note`), the level
    and wall kind, and its bounding box (x_min, y_min, x_max, y_max) in the world frame."""

    name: str
    level: str
    kind: str
    bbox: tuple[float, float, float, float]
    note: str = ""

    KEYS: typing.ClassVar[tuple[str, ...]] = ("name", "level", "kind", "bbox", "note")

    @classmethod
    def parse(cls, entry: object, where: str = "") -> RemovedWall:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}removed_walls: each entry is a mapping ({', '.join(cls.KEYS)}), not {entry!r}")
        bad = sorted(set(entry) - set(cls.KEYS))
        name = str(entry.get("name") or "")
        if bad or not name or not entry.get("kind"):
            raise ValueError(f"{where}removed_walls entry {name or entry!r}: needs name, kind and bbox, may have level and note, not {sorted(entry)}")
        box = entry.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"{where}removed_walls entry {name}: bbox is [x_min, y_min, x_max, y_max], not {box!r}")
        x0, y0, x1, y1 = (float(v) for v in box)
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"{where}removed_walls entry {name}: bbox {box} is empty")
        return cls(name, str(entry.get("level", "0")), str(entry["kind"]), (x0, y0, x1, y1), str(entry.get("note") or ""))

    def matches(self, level: str, wall: Mapping[str, typing.Any]) -> bool:
        """Whether `wall` (a `walls:` entry of `world.yaml`) on `level` is this one."""
        path = wall.get("path") or []
        if str(level) != self.level or wall.get("kind") != self.kind or not wall.get("closed") or len(path) < 3:
            return False
        return all(abs(a - b) <= MATCH_M for a, b in zip(_bbox(path), self.bbox, strict=True))

    def record(self) -> dict[str, typing.Any]:
        return {"name": self.name, "level": self.level, "kind": self.kind, "bbox": list(self.bbox), "note": self.note}


def _bbox(path: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs, ys = [float(p[0]) for p in path], [float(p[1]) for p in path]
    return (min(xs), min(ys), max(xs), max(ys))


def found(doc: Mapping[str, typing.Any], removed: Iterable[RemovedWall]) -> list[tuple[RemovedWall, int, int]]:
    """The walls of `removed` the world (`world.yaml` as data) holds: (wall, level index, wall index), in file order."""
    removed = list(removed)
    hits = []
    for li, level in enumerate(doc.get("levels") or []):
        for wi, wall in enumerate(level.get("walls") or []):
            for entry in removed:
                if entry.matches(str(level.get("id")), wall):
                    hits.append((entry, li, wi))
    return hits


def _load(text: str) -> typing.Any:  # noqa: ANN401 - yaml
    import yaml

    return yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def strip_world_yaml(text: str, walls: Iterable[tuple[int, int]]) -> str:
    """`world.yaml` text without the `walls:` entries (level index, wall index), every other byte as it was. The entries
    must be block sequence items (`- path: ...`), as every world writer here writes them. ValueError when the result
    does not read back as the same world without those walls."""
    import yaml

    walls = sorted(set(walls))
    if not walls:
        return text
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    cuts: list[tuple[int, int]] = []
    for li, wi in walls:
        sequence = _value(_value(root, "levels").value[li], "walls")
        if sequence.flow_style:
            raise ValueError("the walls are a flow sequence ([...]): only block items (- path: ...) can be cut")
        items = sequence.value
        start = text.rfind("\n", 0, items[wi].start_mark.index) + 1
        end = text.rfind("\n", 0, items[wi + 1].start_mark.index) + 1 if wi + 1 < len(items) else _line_end(text, items[wi].end_mark.index)
        if text[start:items[wi].start_mark.index].strip() != "-":
            raise ValueError(f"wall {li}/{wi} does not start its own line (- path: ...)")
        cuts.append((start, end))
    out = text
    for start, end in sorted(cuts, reverse=True):
        out = out[:start] + out[end:]
    want = _load(text)
    for li, wi in sorted(walls, reverse=True):
        del want["levels"][li]["walls"][wi]
    if _load(out) != want:
        raise ValueError("the cut world.yaml does not read back as the world without those walls")
    return out


def _value(node: typing.Any, key: str) -> typing.Any:  # noqa: ANN401 - a yaml node
    for k, v in node.value:
        if k.value == key:
            return v
    raise ValueError(f"world.yaml has no {key!r}")


def _line_end(text: str, index: int) -> int:
    """The index after the line break that ends the line `index` lies on (or the text's end); `index` itself when it is a
    line start already."""
    if index == 0 or text[index - 1] == "\n":
        return index
    newline = text.find("\n", index)
    return len(text) if newline < 0 else newline + 1


def clear_map(image: typing.Any, meta: Mapping[str, typing.Any], removed_paths: Sequence[Sequence[Sequence[float]]], kept_paths: Sequence[Sequence[Sequence[float]]]) -> tuple[typing.Any, int]:  # noqa: ANN401 - numpy
    """The map image (numpy, rows top down, grey or RGB) with the pixels of `removed_paths` (wall polylines) set to
    the free value of the floor round them, and how many it set: an occupied pixel whose centre lies within
    `PIXEL_REACH` pixels of a removed path and further than that from every path of `kept_paths`. The free value is the
    one the most free pixels round the removed walls have (within 0.5 m of their bounding boxes, by the map's
    `free_thresh` and `negate`). Pixels that are not occupied are left as they are."""
    import numpy as np
    import shapely

    resolution = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    negate, free_thresh = int(meta.get("negate", 0)), float(meta.get("free_thresh", 0.196))
    grey = image if image.ndim == 2 else image[..., 0]
    occupancy = grey.astype(float) / 255.0 if negate else (255.0 - grey.astype(float)) / 255.0
    height, width = grey.shape
    reach = PIXEL_REACH * resolution
    removed = shapely.MultiLineString([[tuple(q[:2]) for q in p] for p in removed_paths])
    kept = shapely.MultiLineString([[tuple(q[:2]) for q in p] for p in kept_paths if len(p) > 1]) if kept_paths else None
    x0, y0, x1, y1 = removed.bounds
    pad = 0.5

    def col(x: float) -> int:
        return min(max(int(math.floor((x - ox) / resolution)), 0), width - 1)

    def row(y: float) -> int:
        return min(max(height - 1 - int(math.floor((y - oy) / resolution)), 0), height - 1)

    rows, cols = slice(row(y1 + pad), row(y0 - pad) + 1), slice(col(x0 - pad), col(x1 + pad) + 1)
    rr, cc = np.mgrid[rows, cols]
    xs, ys = ox + (cc + 0.5) * resolution, oy + (height - 1 - rr + 0.5) * resolution
    points = shapely.points(xs.ravel(), ys.ravel())
    near = shapely.dwithin(points, removed, reach).reshape(rr.shape)
    if kept is not None:
        near &= ~shapely.dwithin(points, kept, reach).reshape(rr.shape)
    window = occupancy[rows, cols]
    target = near & (window > free_thresh)
    free = ~near & (window < free_thresh)
    out = image.copy()
    if not target.any():
        return out, 0
    patch = out[rows, cols]
    values, counts = np.unique(patch[free].reshape(-1, *patch.shape[2:]), axis=0, return_counts=True)
    if not len(values):
        raise ValueError("no free pixel round the removed walls to take the floor's value from")
    patch[target] = values[int(np.argmax(counts))]
    return out, int(target.sum())


def remove(world: Path, removed: Iterable[RemovedWall], *, dry_run: bool = False) -> dict[str, typing.Any]:
    """Take the walls of `removed` out of the world at `world` in place: `world.yaml` (`strip_world_yaml`) and each
    level's `map/<level id>.png` with its `.yaml` (`clear_map`). What it found and changed (or would, with `dry_run`).
    A world without those walls is left alone."""
    import yaml

    world = Path(world)
    text = (world / "world.yaml").read_text()
    doc = _load(text)
    hits = found(doc, removed)
    report: dict[str, typing.Any] = {"world": str(world), "walls": [{"name": entry.name, "level": doc["levels"][li].get("id"), "index": wi, "bbox": [round(v, 3) for v in _bbox(doc["levels"][li]["walls"][wi]["path"])]} for entry, li, wi in hits], "map_pixels": {}}
    if not hits:
        return report
    cut = strip_world_yaml(text, [(li, wi) for _, li, wi in hits])
    images: dict[Path, typing.Any] = {}
    for li in sorted({li for _, li, _ in hits}):
        level = doc["levels"][li]
        level_id = str(level.get("id"))
        png, meta_path = world / "map" / f"{level_id}.png", world / "map" / f"{level_id}.yaml"
        if not png.is_file() or not meta_path.is_file():
            report["map_pixels"][level_id] = None
            continue
        import numpy as np
        from PIL import Image

        meta = yaml.safe_load(meta_path.read_text()) or {}
        with Image.open(png) as img:
            mode = img.mode
            pixels = np.asarray(img)
        indices = {wi for _, l2, wi in hits if l2 == li}
        walls = level.get("walls") or []
        cleared, n = clear_map(pixels, meta, [_line(walls[wi]) for wi in sorted(indices)], [_line(w) for i, w in enumerate(walls) if i not in indices and w.get("path")])
        report["map_pixels"][level_id] = n
        if n:
            images[png] = Image.fromarray(cleared, mode=mode)
    if not dry_run:
        (world / "world.yaml").write_text(cut)
        for png, img in images.items():
            img.save(png, format="PNG")
    return report


def _line(wall: Mapping[str, typing.Any]) -> list[Sequence[float]]:
    """A `walls:` entry's path as a polyline, closed back to its start where the wall is closed."""
    path = list(wall["path"])
    return [*path, path[0]] if wall.get("closed") and len(path) > 2 else path


def present(world: Path, removed: Iterable[RemovedWall]) -> list[str]:
    """The names of the walls of `removed` the world at `world` still holds."""
    return [entry.name for entry, _, _ in found(_load((Path(world) / "world.yaml").read_text()), removed)]


def copies(site_root: Path) -> list[Path]:
    """The site's world and its layout copies beside it (`<site>__*` in the same worlds root), site first."""
    site_root = Path(site_root)
    return [site_root, *sorted(p for p in site_root.parent.glob(f"{site_root.name}__*") if (p / "world.yaml").is_file())]


__all__ = ["MATCH_M", "PIXEL_REACH", "RemovedWall", "clear_map", "copies", "found", "present", "remove", "strip_world_yaml"]
