from __future__ import annotations

import typing

import attrs

from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.utils.cattrs import Serializable
from arena_simulation_setup.utils.geometry import Position

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.Wall import WallRealization


@attrs.define
class Wall(Serializable):
    start: Position = attrs.field(converter=Position.converter)
    end: Position = attrs.field(converter=Position.converter)
    kind: str = ''
    material: MaterialIdentifier | None = None
    top: float | None = None
    bottom: float | None = None
    height: float | None = None

    async def assets(self) -> WallRealization:
        """
        Get sub-assets that make up the wall.
        """
        from arena_simulation_setup.tree.Wall import WallDescription, WallIdentifier

        try:
            if self.kind:
                _description = await WallIdentifier(self.kind).resolve()
            else:
                _description = WallDescription.simple(material=self.material if self.material else None)
            return self._close_to(_description.realize(self.start, self.end))
        except Exception as e:
            import logging
            import traceback

            logging.error(f"Failed to load wall assets for wall from {self.start} to {self.end} of kind '{self.kind}' and material '{self.material}': {e}\n{traceback.format_exc()}")

            return self._close_to(WallDescription.simple(material=Material.default('wall')).realize(self.start, self.end))

    def __iter__(self):
        yield self.start
        yield self.end

    def serialize(self) -> dict:
        ser = attrs.asdict(self)
        ser.pop('top', None)
        ser.pop('bottom', None)
        ser.pop('height', None)
        if self.kind or not self.material:
            ser.pop('material', None)
        if not self.kind:
            ser.pop('kind', None)
        return ser

    def _line_z(self, point: Position) -> float:
        """Height of the wall line, which may slope from start to end, at the foot of a point."""
        dx, dy = self.end.x - self.start.x, self.end.y - self.start.y
        span = dx * dx + dy * dy
        t = ((point.x - self.start.x) * dx + (point.y - self.start.y) * dy) / span if span > 0.0 else 0.0
        return self.start.z + t * (self.end.z - self.start.z)

    def _close_to(self, realization: WallRealization) -> WallRealization:
        """Raise the highest segment of the wall stack to `top` above the wall line, or make the wall one segment from `bottom` to `top`, or cut the stack at `height` and raise its highest segment to it."""
        segments, obstacles = realization
        segments = list(segments)
        top, bottom = self.top, self.bottom
        if not segments or (top is None and self.height is None):
            return segments, obstacles
        if top is not None and bottom is not None:
            segment = max(segments, key=lambda segment: segment.start.z + segment.height)
            lintel = attrs.evolve(
                segment,
                start=Position(segment.start.x, segment.start.y, self._line_z(segment.start) + bottom),
                end=Position(segment.end.x, segment.end.y, self._line_z(segment.end) + bottom),
                height=top - bottom,
            )
            return [lintel], ()
        cut = self.height
        goal = typing.cast(float, cut if top is None else top)
        kept = []
        for segment in segments:
            base = segment.start.z - self._line_z(segment.start)
            if cut is None:
                kept.append((base, segment))
            elif base < cut - 1e-9:
                kept.append((base, attrs.evolve(segment, height=min(segment.height, cut - base))))
        if not kept:
            return [], obstacles
        highest = max(range(len(kept)), key=lambda index: kept[index][0] + kept[index][1].height)
        base, segment = kept[highest]
        if base + segment.height < goal:
            kept[highest] = (base, attrs.evolve(segment, height=goal - base))
        return [segment for _base, segment in kept], obstacles
