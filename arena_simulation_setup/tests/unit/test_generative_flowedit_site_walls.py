from __future__ import annotations

import difflib
import os
from pathlib import Path

import attrs
import pytest
import yaml

pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("shapely")

from arena_simulation_setup.generative.flowedit import cli, infra  # noqa: E402
from arena_simulation_setup.generative.flowedit import site_walls as SW  # noqa: E402

#: the twin's temporary sign (smu_link_full_v2, level 0) and the two scanned pillars beside it, which stay
SIGN_BBOX = (152.03, 16.67, 152.53, 17.25)
SIGN = [[152.17, 17.182], [152.25, 17.223], [152.133, 17.25], [152.032, 16.866], [152.254, 16.672], [152.45, 16.792], [152.529, 17.105], [152.36, 16.954], [152.218, 17.01]]
PILLAR_NORTH = [[150.55, 17.9], [150.9, 17.51], [151.49, 17.8], [151.3, 18.33], [150.7, 18.25]]
PILLAR_SOUTH = [[150.47, 12.3], [150.9, 11.91], [151.37, 12.2], [151.1, 12.65], [150.6, 12.6]]
PILLAR_BBOXES = {"north": (150.55, 17.51, 151.49, 18.33), "south": (150.47, 11.91, 151.37, 12.65)}
#: the walkway round them, and the east wall the sign left a 0.56 m slot to
OUTER = [[149.0, 10.0], [153.07, 10.0], [153.07, 20.0], [149.0, 20.0]]
EAST_WALL = [[153.07, 10.5], [153.07, 19.5]]
ORIGIN, RESOLUTION = (148.0, 9.0), 0.05
REMOVED = SW.RemovedWall("wet_floor_sign", "0", "scan_boundary", SIGN_BBOX)


def _walls(order: str) -> list[dict]:
    """The walls in the order `order` names them: o(uter), s(ign), n(orth pillar), p (south pillar), e(ast wall)."""
    walls = {
        "o": {"path": OUTER, "closed": True, "kind": "scan_boundary"},
        "s": {"path": SIGN, "closed": True, "kind": "scan_boundary"},
        "n": {"path": PILLAR_NORTH, "closed": True, "kind": "scan_boundary"},
        "p": {"path": PILLAR_SOUTH, "closed": True, "kind": "scan_boundary"},
        "e": {"path": EAST_WALL, "closed": False, "kind": "scan_boundary"},
    }
    return [walls[key] for key in order]


def _world(root: Path, order: str = "osnpe", *, flow: bool = False) -> Path:
    """A world at `root` like the twin's corner by the loading point: `world.yaml` block style as the site's (or with
    flow-style paths as a layout copy's, `flow`), and its map at 0.05 m, the walls drawn as 1-pixel lines on free floor."""
    import numpy as np
    from PIL import Image, ImageDraw

    root.mkdir(parents=True)
    doc = {"format": "arena-world/2", "defaults": {"floor": False, "ceiling": False}, "levels": [{"id": "0", "zones": [{"name": "lower_concourse", "polygon": OUTER}], "walls": _walls(order), "objects": [{"name": "link_scan", "model": "smu_link_full", "pose": [85.107, 157.336, 0.0, 0.0, 0.0, 0.0]}]}]}
    if flow:

        class Dumper(yaml.SafeDumper):
            pass

        def points(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
            flat = all(isinstance(v, (int, float)) for v in data) or all(isinstance(v, list) and all(isinstance(q, (int, float)) for q in v) for v in data)
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flat and bool(data))

        Dumper.add_representer(list, points)
        text = yaml.dump(doc, Dumper=Dumper, sort_keys=False)
    else:
        text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    (root / "world.yaml").write_text(text)
    width, height = 120, 240
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    def px(p: list[float]) -> tuple[float, float]:
        return ((p[0] - ORIGIN[0]) / RESOLUTION, height - (p[1] - ORIGIN[1]) / RESOLUTION)

    draw.polygon([px(p) for p in OUTER], fill=(255, 255, 255))
    for wall in _walls(order):
        line = [px(p) for p in wall["path"]] + ([px(wall["path"][0])] if wall["closed"] else [])
        draw.line(line, fill=(0, 0, 0), width=1)
    (root / "map").mkdir()
    img.save(root / "map" / "0.png")
    (root / "map" / "0.yaml").write_text(yaml.safe_dump({"free_thresh": 0.1, "image": "0.png", "negate": 0, "occupied_thresh": 0.9, "origin": [*ORIGIN, 0], "resolution": RESOLUTION}))
    assert np.asarray(img).shape == (height, width, 3)
    return root


def _pixels(root: Path) -> object:
    import numpy as np
    from PIL import Image

    with Image.open(root / "map" / "0.png") as img:
        return np.asarray(img).copy()


def _occupied_in(pixels: object, box: tuple[float, float, float, float], pad: float = 0.1) -> int:
    height = pixels.shape[0]
    c0, c1 = int((box[0] - pad - ORIGIN[0]) / RESOLUTION), int((box[2] + pad - ORIGIN[0]) / RESOLUTION)
    r0, r1 = height - 1 - int((box[3] + pad - ORIGIN[1]) / RESOLUTION), height - 1 - int((box[1] - pad - ORIGIN[1]) / RESOLUTION)
    return int((pixels[r0 : r1 + 1, c0 : c1 + 1, 0] < 128).sum())


@pytest.mark.parametrize(("order", "flow"), [("osnpe", False), ("onpse", True), ("snpeo", False)])
def test_remove_takes_the_sign_out_of_world_yaml_and_the_map_by_its_geometry_whatever_its_index(tmp_path: Path, order: str, flow: bool) -> None:
    import numpy as np

    root = _world(tmp_path / "w", order, flow=flow)
    before_text, before_pixels = (root / "world.yaml").read_text(), _pixels(root)
    assert _occupied_in(before_pixels, SIGN_BBOX) > 20
    report = SW.remove(root, [REMOVED])
    assert [(w["name"], w["index"]) for w in report["walls"]] == [("wet_floor_sign", order.index("s"))]
    # world.yaml: the same world without that wall; every other line as it was
    after = yaml.safe_load((root / "world.yaml").read_text())
    want = yaml.safe_load(before_text)
    del want["levels"][0]["walls"][order.index("s")]
    assert after == want
    # one block of lines cut, nothing added or changed
    ops = difflib.SequenceMatcher(None, before_text.splitlines(), (root / "world.yaml").read_text().splitlines(), autojunk=False).get_opcodes()
    assert [op for op, *_ in ops if op != "equal"] == ["delete"]
    # the map: no occupied pixel left round the sign, the cleared pixels take the floor's value, nothing else changed
    after_pixels = _pixels(root)
    assert _occupied_in(after_pixels, SIGN_BBOX) == 0
    changed = (after_pixels != before_pixels).any(axis=-1)
    assert int(changed.sum()) == report["map_pixels"]["0"] > 20
    assert (after_pixels[changed] == 255).all()
    rows, cols = np.nonzero(changed)
    xs, ys = ORIGIN[0] + (cols + 0.5) * RESOLUTION, ORIGIN[1] + (after_pixels.shape[0] - 1 - rows + 0.5) * RESOLUTION
    assert xs.min() >= SIGN_BBOX[0] - 0.1 and xs.max() <= SIGN_BBOX[2] + 0.1 and ys.min() >= SIGN_BBOX[1] - 0.1 and ys.max() <= SIGN_BBOX[3] + 0.1
    # the pillars and the east wall stay, in world.yaml and in the map
    for name, box in PILLAR_BBOXES.items():
        assert any(SW.RemovedWall(name, "0", "scan_boundary", box).matches("0", w) for w in after["levels"][0]["walls"]), name
        assert _occupied_in(after_pixels, box, pad=0.0) == _occupied_in(before_pixels, box, pad=0.0) > 0
    assert _occupied_in(after_pixels, (153.0, 10.6, 153.1, 19.4), pad=0.0) == _occupied_in(before_pixels, (153.0, 10.6, 153.1, 19.4), pad=0.0) > 0
    # a second run finds nothing and leaves the files alone
    text, pixels = (root / "world.yaml").read_text(), _pixels(root)
    assert SW.remove(root, [REMOVED]) == {"world": str(root), "walls": [], "map_pixels": {}}
    assert (root / "world.yaml").read_text() == text and (_pixels(root) == pixels).all()
    assert SW.present(root, [REMOVED]) == []


def test_remove_with_dry_run_reports_and_changes_nothing(tmp_path: Path) -> None:
    root = _world(tmp_path / "w")
    text, png = (root / "world.yaml").read_text(), (root / "map" / "0.png").read_bytes()
    report = SW.remove(root, [REMOVED], dry_run=True)
    assert [w["index"] for w in report["walls"]] == [1] and report["map_pixels"]["0"] > 20
    assert (root / "world.yaml").read_text() == text and (root / "map" / "0.png").read_bytes() == png
    assert SW.present(root, [REMOVED]) == ["wet_floor_sign"]


@pytest.mark.parametrize(
    "wall",
    [
        {"path": [[x + 0.1, y] for x, y in SIGN], "closed": True, "kind": "scan_boundary"},  # 0.1 m off
        {"path": SIGN, "closed": False, "kind": "scan_boundary"},  # open
        {"path": SIGN, "closed": True, "kind": "plaster"},  # another kind
    ],
)
def test_a_wall_matches_only_by_level_kind_closed_loop_and_bounding_box_within_5_cm(wall: dict) -> None:
    assert REMOVED.matches("0", {"path": SIGN, "closed": True, "kind": "scan_boundary"})
    assert REMOVED.matches("0", {"path": [[x + 0.04, y - 0.04] for x, y in SIGN], "closed": True, "kind": "scan_boundary"})
    assert not REMOVED.matches("1", {"path": SIGN, "closed": True, "kind": "scan_boundary"})
    assert not REMOVED.matches("0", wall)


def test_the_v2_twins_site_config_removes_the_sign_and_records_it() -> None:
    config = infra.site_config("smu_link_full_v2")
    (sign,) = config.removed_walls
    assert (sign.name, sign.level, sign.kind, sign.bbox) == ("wet_floor_sign", "0", "scan_boundary", SIGN_BBOX)
    assert config.record()["removed_walls"] == [sign.record()]
    assert "removed_walls" not in infra.site_config("smu_link_full").record()


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"name": "x", "kind": "scan_boundary", "bbox": [1, 2, 3]}, "bbox is"),
        ({"name": "x", "kind": "scan_boundary", "bbox": [3, 2, 1, 4]}, "is empty"),
        ({"kind": "scan_boundary", "bbox": [1, 2, 3, 4]}, "needs name"),
        ({"name": "x", "kind": "scan_boundary", "bbox": [1, 2, 3, 4], "index": 8}, "needs name"),
    ],
)
def test_a_site_config_refuses_a_removed_wall_it_cannot_match(entry: dict, match: str) -> None:
    doc = yaml.safe_load(infra.site_config("smu_link_full_v2").path.read_text())
    doc["removed_walls"] = [entry]
    with pytest.raises(ValueError, match=match):
        infra.SiteConfig.parse(doc)


def _sign_only_config(tmp_path: Path) -> Path:
    """The v2 twin's site config with its removed walls only (the test worlds lack the stairs' outlines)."""
    doc = yaml.safe_load(infra.site_config("smu_link_full_v2").path.read_text())
    doc.pop("added_walls")
    path = tmp_path / "smu_link_full_v2.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def test_remove_walls_cleans_the_site_and_its_layout_copies_and_the_writers_refuse_a_site_that_holds_the_sign(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = tmp_path / "worlds"
    site = _world(worlds / "smu_link_full_v2")
    copies = [_world(worlds / "smu_link_full_v2__layout_2242008c", "onpse", flow=True), _world(worlds / "smu_link_full_v2__layout_4b5e5e6f")]
    other = _world(worlds / "smu_link_full")
    path = _sign_only_config(tmp_path)
    config = infra.site_config("smu_link_full_v2", path)
    with pytest.raises(ValueError, match=r"still holds the walls \['wet_floor_sign'\].*site-walls"):
        config.check_site(site)
    assert cli.main(["infra", "site-walls", "--site", str(site), "--site-config", str(path), "--dry-run"]) == 0
    assert all(SW.present(w, config.removed_walls) == ["wet_floor_sign"] for w in (site, *copies))
    # remove-walls is the same command
    assert cli.main(["infra", "remove-walls", "--site", str(site), "--site-config", str(path)]) == 0
    out = capsys.readouterr().out
    assert all(f"{w}: removed wet_floor_sign (level 0 wall " in out for w in (site, *copies))
    assert all(SW.present(w, config.removed_walls) == [] and _occupied_in(_pixels(w), SIGN_BBOX) == 0 for w in (site, *copies))
    config.check_site(site)
    # another world beside it is no copy of the site: left alone
    assert SW.present(other, config.removed_walls) == ["wet_floor_sign"]
    assert cli.main(["infra", "site-walls", "--site", str(site), "--site-config", str(path)]) == 0
    assert "none left" in capsys.readouterr().out


#: the T-junction stairs' foot: the outer wall coming up x 131.45 and turning west at y 48.07 (walls[0]), the deck's
#: outline from its corner (131.55, 48.40) east (walls[37]); the slot between them joins the lower concourse (south-east)
#: to the corridor above the stairs (north-west)
STAIR_OUTER = [[131.45, 44.0], [131.45, 48.07], [131.203, 48.07], [131.229, 48.177], [131.156, 48.25], [130.87, 48.31], [130.0, 48.31]]
STAIR_DECK = [[131.55, 48.4], [132.382, 48.35], [134.49, 48.45], [134.49, 49.0], [131.65, 49.0], [131.65, 48.45], [131.6, 48.52]]
SLOT = SW.AddedWall("stair_foot_slot", "0", "scan_boundary", ((131.45, 48.07), (131.55, 48.40)))
LOWER, ABOVE = (131.9, 47.9), (131.35, 48.75)
#: the window round the slot the map check looks in (the test world is open round the two outlines)
NEAR_SLOT = (130.9, 47.6, 132.2, 48.95)


def _stairs(root: Path, *, flow: bool = False) -> Path:
    """A world of the stairs' foot: the floor round it free, the two outlines drawn, the slot open."""
    import numpy as np
    from PIL import Image, ImageDraw

    root.mkdir(parents=True)
    walls = [{"path": STAIR_OUTER, "closed": False, "kind": "scan_boundary"}, {"path": STAIR_DECK, "closed": True, "kind": "scan_boundary"}]
    doc = {"format": "arena-world/2", "levels": [{"id": "0", "zones": [], "walls": walls, "objects": []}]}
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    if flow:
        text = text.replace("- - ", "- [").replace("\n      - ", ", ")
        text = "\n".join(line + "]" if line.lstrip().startswith("- [") else line for line in text.split("\n"))
        assert yaml.safe_load(text) == doc
    (root / "world.yaml").write_text(text)
    origin, width, height = (128.0, 45.0), 160, 120
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    def px(p: list[float]) -> tuple[float, float]:
        return ((p[0] - origin[0]) / RESOLUTION, height - (p[1] - origin[1]) / RESOLUTION)

    for wall in walls:
        draw.line([px(p) for p in wall["path"]] + ([px(wall["path"][0])] if wall["closed"] else []), fill=0, width=1)
    (root / "map").mkdir()
    img.save(root / "map" / "0.png")
    (root / "map" / "0.yaml").write_text(yaml.safe_dump({"free_thresh": 0.1, "image": "0.png", "negate": 0, "occupied_thresh": 0.9, "origin": [*origin, 0], "resolution": RESOLUTION}))
    assert np.asarray(img).shape == (height, width)
    return root


def _connected(root: Path, a: tuple[float, float], b: tuple[float, float], box: tuple[float, float, float, float] | None = None) -> bool:
    """Whether free pixels of the map join `a` and `b`, 8-connected (a diagonal step counts), within `box` (x_min, y_min,
    x_max, y_max) only when given."""
    from collections import deque

    import numpy as np
    from PIL import Image

    meta = yaml.safe_load((root / "map" / "0.yaml").read_text())
    with Image.open(root / "map" / "0.png") as img:
        free = np.asarray(img.convert("L")) > 128
    (ox, oy), res, height = meta["origin"][:2], meta["resolution"], free.shape[0]

    def cell(p: tuple[float, float]) -> tuple[int, int]:
        return height - 1 - int((p[1] - oy) / res), int((p[0] - ox) / res)

    r0, c0, r1, c1 = 0, 0, free.shape[0] - 1, free.shape[1] - 1
    if box is not None:
        (r1, c0), (r0, c1) = cell((box[0], box[1])), cell((box[2], box[3]))
    start, goal = cell(a), cell(b)
    assert free[start] and free[goal]
    seen, todo = {start}, deque([start])
    while todo:
        r, c = todo.popleft()
        if (r, c) == goal:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr, c + dc)
                if r0 <= n[0] <= r1 and c0 <= n[1] <= c1 and free[n] and n not in seen:
                    seen.add(n)
                    todo.append(n)
    return False


@pytest.mark.parametrize("flow", [False, True])
def test_add_closes_the_stair_foot_slot_in_world_yaml_and_the_map_for_robots(tmp_path: Path, flow: bool) -> None:
    root = _stairs(tmp_path / "w", flow=flow)
    before = (root / "world.yaml").read_text()
    assert _connected(root, LOWER, ABOVE, NEAR_SLOT)
    assert SW.missing(root, [SLOT]) == ["stair_foot_slot"]
    report = SW.add(root, [SLOT])
    assert [(w["name"], w["index"]) for w in report["added"]] == [("stair_foot_slot", 2)] and report["map_pixels"]["0"] > 3
    # world.yaml: the same world with the wall appended; every line before it as it was
    after = (root / "world.yaml").read_text()
    want = yaml.safe_load(before)
    want["levels"][0]["walls"].append({"path": [[131.45, 48.07], [131.55, 48.4]], "closed": False, "kind": "scan_boundary"})
    assert yaml.safe_load(after) == want
    ops = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False).get_opcodes()
    assert [op for op, *_ in ops if op != "equal"] == ["insert"]
    # the map: no free path from the lower concourse to the corridor above the stairs any more
    assert not _connected(root, LOWER, ABOVE, NEAR_SLOT)
    assert SW.missing(root, [SLOT]) == []
    # a second run adds nothing
    pixels = _pixels(root)
    assert SW.add(root, [SLOT])["added"] == [] and (root / "world.yaml").read_text() == after and (_pixels(root) == pixels).all()


def test_add_refuses_a_wall_whose_end_is_off_every_other_wall(tmp_path: Path) -> None:
    root = _stairs(tmp_path / "w")
    short = SW.AddedWall("short", "0", "scan_boundary", ((131.45, 48.07), (131.52, 48.30)))
    with pytest.raises(ValueError, match="would not close the gap"):
        SW.add(root, [short])
    assert SW.missing(root, [short]) == ["short"]


def test_the_v2_twins_site_config_closes_the_stair_foot_slot_and_the_writers_refuse_a_site_without_it(tmp_path: Path) -> None:
    config = infra.site_config("smu_link_full_v2")
    (slot,) = config.added_walls
    assert slot == attrs.evolve(SLOT, note=slot.note) and not slot.closed
    assert config.record()["added_walls"] == [slot.record()]
    root = _stairs(tmp_path / "w")
    with pytest.raises(ValueError, match=r"lacks the walls \['stair_foot_slot'\]"):
        attrs.evolve(config, removed_walls=()).check_site(root)
    SW.add(root, config.added_walls)
    attrs.evolve(config, removed_walls=()).check_site(root)


def test_every_installed_v2_twin_world_is_free_of_the_sign_and_keeps_the_pillars() -> None:
    """The local worlds smu_link_full_v2 and its copies (`<site>__*`), when installed: no wall of the sign in world.yaml,
    no occupied pixel round it in map/0.png, and the two pillars beside it still there; the stair foot slot closed in
    world.yaml, and no free map path through it from the lower concourse to the corridor above the stairs."""
    import numpy as np
    from PIL import Image

    root = Path(os.environ.get("ARENA_DIR", "/nonexistent")) / "arena_simulation_setup" / "assets" / "worlds" / "smu_link_full_v2"
    if not (root / "world.yaml").is_file():
        pytest.skip("the local world smu_link_full_v2 is not installed")
    config = infra.site_config("smu_link_full_v2")
    pillars = [SW.RemovedWall(name, "0", "scan_boundary", box) for name, box in PILLAR_BBOXES.items()]
    problems = []
    for world in SW.copies(root):
        if left := SW.present(world, config.removed_walls):
            problems.append(f"{world.name}: world.yaml holds {left}")
        if missing := sorted({p.name for p in pillars} - set(SW.present(world, pillars))):
            problems.append(f"{world.name}: no pillar {missing}")
        if lacking := SW.missing(world, config.added_walls):
            problems.append(f"{world.name}: world.yaml lacks {lacking}")
        elif _connected(world, (131.9, 47.9), (131.0, 50.3), (128.0, 45.0, 136.0, 52.0)):
            problems.append(f"{world.name}: free map pixels join the lower concourse to the corridor above the T-junction stairs")
        meta = yaml.safe_load((world / "map" / "0.yaml").read_text())
        with Image.open(world / "map" / "0.png") as img:
            grey = np.asarray(img.convert("L"))
        (ox, oy), res, height = meta["origin"][:2], float(meta["resolution"]), grey.shape[0]
        box = [v + d for v, d in zip(SIGN_BBOX, (-0.1, -0.1, 0.1, 0.1), strict=True)]
        cols = slice(int((box[0] - ox) / res), int((box[2] - ox) / res) + 1)
        rows = slice(height - 1 - int((box[3] - oy) / res), height - int((box[1] - oy) / res))
        if (occupied := int((grey[rows, cols] < 128).sum())) > 0:
            problems.append(f"{world.name}: {occupied} occupied map pixels round the sign")
    assert not problems, problems
