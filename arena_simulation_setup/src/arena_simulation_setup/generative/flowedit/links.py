"""The stairs and escalators walkers take, in HumanSim's form, for every flowedit measure that walks.

A site's walkers change floor over two kinds of link, both handed to HumanSim as teleport zones: the halves a site
config names (`SiteConfig.links`, for a world that lays its levels out side by side and draws no stairs), and the
halves of the world's own staircases (`Level.stairs`: stairs and escalators; a ramp is floor). `world_links` builds the
second kind with the builder the runtime hands HumanSim its zones from (`Stairs.teleport_zone`): entry at the mouth
facing up the half, the hold at the seam trigger, the footprint out to the reach past the seam, the destination the
half's `to` partner, a belt on a running escalator half. `site_links` puts both together. They are what the walking distances
(`demand.Walks`), the elements' keep-out at stair mouths (`infra.link_keep_out`) and the crowd's routes of the walkway
rule (`walkway.Routes`) walk over; wheelchair users never take them. A world's stairs are never written into a
crowd scenario: the runtime hands them to HumanSim itself."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.tree.World.model import Level

    from .edit import Site

#: The floor in front of a link's mouth that no element takes, out from its entry line across its width: walkers step
#: off a stair or escalator half there and queue for it [m]
LINK_APRON_M = 1.5
#: A half's entry is on a walking floor where the floor comes this near it: the floors keep their clearance from the
#: stair's own walls, so a mouth on the floor lies up to about a metre off its nearest walkable cell [m]
MOUTH_SNAP_M = 1.5
#: Halves whose entries lie off the walking floor and this near each other share a landing between two flights [m]
LANDING_M = 3.0
#: How fast walkers climb a still stair and cross its landing (HumanSim's `teleport_zones.STAIR_SPEED`) [m/s]
STAIR_SPEED_MPS = 0.7


def world_links(level: Level) -> list[dict[str, Any]]:
    """The stair and escalator halves of `level` as HumanSim teleport zones, each built by `Stairs.teleport_zone`, the
    builder the runtime hands the crowd simulator its zones from, here with every escalator running; a half whose `to`
    partner is not on this level leads nowhere from it (`destinations` empty), as the runtime leaves it."""
    halves = [s for s in level.stairs if s.kind != "ramp"]
    names = {s.name for s in halves}

    def pose(p: tuple[float, float, float]) -> dict[str, float]:
        return {"x": round(p[0], 4), "y": round(p[1], 4), "theta": round(p[2], 6)}

    out = []
    for s in halves:
        zone = s.teleport_zone(names)
        out.append(
            {
                "name": zone.name,
                "kind": zone.kind,
                "destinations": list(zone.destinations),
                "entry": pose(zone.entry),
                "wait": pose(zone.wait),
                "hold": pose(zone.hold),
                "footprint": [{"x": round(x, 4), "y": round(y, 4)} for x, y in zone.footprint],
                "delay": zone.delay,
                "belt": float(zone.belt),
            }
        )
    return out


def site_links(site: Site, level_id: str, config_links: Iterable[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """Every link walkers take on `level_id` of `site`: the site config's `config_links` and the world's own stair and
    escalator halves (`world_links`). ValueError when a world half and a config link share a name."""
    config = [dict(z) for z in config_links]
    world = world_links(site.world.levels[level_id])
    clash = sorted({z["name"] for z in world} & {str(z.get("name")) for z in config})
    if clash:
        raise ValueError(f"links {clash} are both world stairs and site config links: a world that draws its stairs needs no config links for them")
    return [*config, *world]


def _point(p: Mapping[str, Any]) -> tuple[float, float]:
    return (float(p["x"]), float(p["y"]))


def _run(zone: Mapping[str, Any]) -> float:
    """A half's stretch from its entry to its hold [m] (none without a hold)."""
    return math.dist(_point(zone["entry"]), _point(zone["hold"])) if zone.get("hold") else 0.0


def _over(zone: Mapping[str, Any], metres: float) -> float:
    """Seconds over `metres` of a half: at its belt's speed, else at `STAIR_SPEED_MPS`."""
    belt = abs(float(zone.get("belt") or 0.0))
    return metres / (belt if belt > 0.0 else STAIR_SPEED_MPS)


def floor_links(zones: Iterable[Mapping[str, Any]], on_floor: Callable[[tuple[float, float]], bool]) -> list[dict[str, Any]]:
    """`zones` (HumanSim teleport zones) as the links a measure on a walking floor walks over: from floor to floor. A
    half whose entry lies off the floor (`on_floor` false) stands on a landing between two flights (the world draws the
    landing as stair, no zone covers it): a walker landing there crosses to the other halves whose entries lie off the
    floor within `LANDING_M` of it and goes on. Each way from a half on the floor over such landings to a half on the
    floor becomes one link: the boarding half with that destination, its `delay` the landing halves' stretches and the
    walks across the landings in seconds (at the halves' belts, else `STAIR_SPEED_MPS`). Halves on the floor keep their
    direct destinations; halves off it are left out. ValueError for a half on the floor whose way ends on a landing with
    no way on: no measure could take it."""
    zones = [dict(z) for z in zones]
    by_name = {str(z["name"]): z for z in zones}
    off = {name for name, z in by_name.items() if not on_floor(_point(z["entry"]))}
    landing = {a: [b for b in off if b != a and math.dist(_point(by_name[a]["entry"]), _point(by_name[b]["entry"])) <= LANDING_M] for a in off}
    out: list[dict[str, Any]] = []
    for name, z in by_name.items():
        if name in off:
            continue
        direct = [d for d in z.get("destinations") or [] if d in by_name and d not in off]
        out.append({**z, "destinations": direct})
        for first in (d for d in z.get("destinations") or [] if d in off):
            # walk on over landings: (the half landed on, seconds since the boarding half's hold, halves seen)
            frontier = [(first, _over(by_name[first], _run(by_name[first])), {name, first})]
            reached = False
            while frontier:
                here, seconds, seen = frontier.pop()
                for across in landing[here]:
                    if across in seen:
                        continue
                    walk = math.dist(_point(by_name[here]["entry"]), _point(by_name[across]["entry"])) / STAIR_SPEED_MPS
                    board = _over(by_name[across], _run(by_name[across]))
                    for nxt in by_name[across].get("destinations") or []:
                        if nxt not in by_name or nxt in seen:
                            continue
                        if nxt in off:
                            frontier.append((nxt, seconds + walk + board + _over(by_name[nxt], _run(by_name[nxt])), seen | {across, nxt}))
                        else:
                            reached = True
                            out.append({**z, "name": f"{name}>{across}>{nxt}", "destinations": [nxt], "delay": round(float(z.get("delay") or 0.0) + seconds + walk + board, 3)})
            if not reached:
                raise ValueError(f"link {name!r} leads to {first!r}, whose entry lies off the walking floor with no half within {LANDING_M:g} m to go on by: no landing joins it to the floor")
    return out


def link_keep_out(links: Iterable[Mapping[str, Any]]) -> list[shapely.Polygon]:
    """The floor of each link with a footprint (teleport zones in HumanSim's form: stair and escalator halves, cabins) and
    an apron in front of its mouth, `LINK_APRON_M` out from its entry line and as wide as its footprint across it: an
    element there would stand on the stair, or in the way of the walkers stepping off it and queueing for it."""
    import shapely

    out = []
    for link in links:
        points = [(float(p["x"]), float(p["y"])) for p in link.get("footprint") or []]
        if len(points) < 3:
            continue
        entry = link["entry"]
        ex, ey = float(entry["x"]), float(entry["y"])
        ax, ay = math.cos(float(entry.get("theta", 0.0))), math.sin(float(entry.get("theta", 0.0)))
        across = [-(x - ex) * ay + (y - ey) * ax for x, y in points]
        lo, hi = min(across), max(across)
        out.append(shapely.Polygon(points))
        out.append(shapely.Polygon([(ex - along * ax - c * ay, ey - along * ay + c * ax) for along, c in ((0.0, lo), (LINK_APRON_M, lo), (LINK_APRON_M, hi), (0.0, hi))]))
    return out


__all__ = ["LANDING_M", "LINK_APRON_M", "MOUTH_SNAP_M", "floor_links", "link_keep_out", "site_links", "world_links"]
