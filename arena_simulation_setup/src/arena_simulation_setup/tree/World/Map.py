import io
import logging
import math
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import shapely
import shapely.affinity
import yaml

from arena_simulation_setup.tree import PathView

PREVIEW_PIXEL_BUDGET = 800
PREVIEW_MIN_RESOLUTION = 0.05


def preview_resolution(size_m: float) -> float:
    """Meters per pixel that fit `size_m` into PREVIEW_PIXEL_BUDGET pixels, never finer than PREVIEW_MIN_RESOLUTION."""
    return max(PREVIEW_MIN_RESOLUTION, size_m / PREVIEW_PIXEL_BUDGET)


class PixelFrame(typing.NamedTuple):
    """How `Map` maps level-frame meters to image pixels: the affine `to_pixels` (y flipped), the image `size` (width,
    height) in pixels, the `padding` in pixels and the level-frame `origin` of the image's bottom-left corner."""

    to_pixels: list[float]
    size: tuple[int, int]
    padding: int
    origin: tuple[float, float]

    def polygon_pixels(self, polygon: shapely.Geometry) -> list[tuple[int, int]] | None:
        """The integer pixel ring `Map` fills for a level-frame polygon, None when it has under 3 points left."""
        poly = _clean(shapely.affinity.affine_transform(polygon, self.to_pixels))
        assert isinstance(poly, shapely.Polygon)
        if len(poly.exterior.coords) < 3:
            return None
        return _as_int(poly.exterior.coords, self.padding)

    def polygon_mask(self, polygon: shapely.Geometry) -> tuple[int, int, np.ndarray] | None:
        """The pixels `Map` fills for a level-frame static-object polygon, as the (row, col) of the top-left corner and a
        bool mask, clipped to the image; None when nothing of it lands on the image. Pixel for pixel what drawing the
        polygon into the whole image fills, at the cost of its bounding box."""
        ring = self.polygon_pixels(polygon)
        if ring is None:
            return None
        xs, ys = [x for x, _ in ring], [y for _, y in ring]
        c0, r0 = min(xs), min(ys)
        local = PIL.Image.new('1', (max(xs) - c0 + 1, max(ys) - r0 + 1), 0)
        PIL.ImageDraw.Draw(local).polygon([(x - c0, y - r0) for x, y in ring], fill=1)
        mask = np.asarray(local, dtype=bool)
        width, height = self.size
        rows, cols = slice(max(0, -r0), max(0, min(mask.shape[0], height - r0))), slice(max(0, -c0), max(0, min(mask.shape[1], width - c0)))
        mask = mask[rows, cols]
        if not mask.any():
            return None
        return r0 + rows.start, c0 + cols.start, mask


def _clean(shape: shapely.Geometry) -> shapely.Geometry:
    """Cleanup stays per part: snapping a whole collection at once lets neighbouring
    zones merge back into one ring, and a filled exterior would swallow the island."""
    shape = shapely.set_precision(shape, 0.01)
    shape = shapely.make_valid(shape)
    shape = shapely.remove_repeated_points(shape)
    return shape


def _as_int(coords: Iterable[Sequence[float]], padding: int) -> list[tuple[int, int]]:
    return [(int(math.trunc(x) + padding), int(math.trunc(y) + padding)) for (x, y, *_) in coords]


class Map(PathView):
    @property
    def map_yaml(self) -> Path:
        return self.path / 'map.yaml'

    @property
    def map_png(self) -> Path:
        return self.path / 'map.png'

    @classmethod
    def _render_image(
        cls,
        rooms: shapely.MultiPolygon,
        doors: shapely.MultiPolygon,
        walls: shapely.MultiLineString,
        resolution: float = 0.01,
        padding: int = 5,
        *,
        static_objects: Iterable[tuple[str, shapely.Polygon]] = (),
        asset_color: str | None = "grey",
        asset_name_color: str | None = "blue",
        cabins: shapely.MultiPolygon | None = None,
    ) -> tuple[PIL.Image.Image, tuple[float, float]]:
        frame = cls.pixel_frame(rooms, doors, resolution=resolution, padding=padding, cabins=cabins)
        cabins = cabins if cabins is not None else shapely.MultiPolygon()
        img = PIL.Image.new('RGB', frame.size, color='black')
        to_pixels = frame.to_pixels

        def to_pixel_parts(shape: shapely.Geometry) -> np.ndarray:
            return shapely.get_parts(shapely.affinity.affine_transform(shape, to_pixels))

        tf = _clean

        def as_int(coords: Iterable[Sequence[float]]) -> list[tuple[int, int]]:
            return _as_int(coords, padding)

        draw = PIL.ImageDraw.Draw(img)
        for cutout in to_pixel_parts(rooms):
            poly = tf(cutout)
            assert isinstance(poly, shapely.Polygon)
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        for cutout in to_pixel_parts(cabins):
            poly = tf(cutout)
            assert isinstance(poly, shapely.Polygon)
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        for wall in to_pixel_parts(walls):
            line = tf(wall)
            assert isinstance(line, shapely.LineString)
            draw.line(as_int(line.coords), fill='black', width=1)

        for cutout in to_pixel_parts(doors):
            poly = tf(cutout)
            assert isinstance(poly, shapely.Polygon)
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        if asset_color is not None:
            for name, obj in static_objects:
                logging.debug(f"Drawing asset '{name}' with geometry: {obj} in color {asset_color}")
                ring = frame.polygon_pixels(obj)
                if ring is None:
                    logging.warning(f"Skipping asset '{name}' because it has insufficient geometry to draw (under 3 coordinates).")
                    continue
                draw.polygon(ring, fill=asset_color)
                if asset_name_color is not None:
                    _max_x, _max_y = max(x for x, _ in ring) - padding, max(y for _, y in ring) - padding
                    logging.debug(f"Drawing name for asset '{name}' at ({_max_x}, {_max_y}) color {asset_name_color}")
                    draw.text((_max_x, _max_y), name, fill=asset_name_color)

        return img, frame.origin

    @classmethod
    def pixel_frame(
        cls,
        rooms: shapely.MultiPolygon,
        doors: shapely.MultiPolygon,
        walls: shapely.MultiLineString | None = None,
        resolution: float = 0.01,
        padding: int = 5,
        *,
        cabins: shapely.MultiPolygon | None = None,
        **_drawn: object,
    ) -> PixelFrame:
        """The pixel frame `rasterize` and `generate_png` draw in: the rooms', doors' and cabins' bounds plus `padding`
        pixels. Takes the same keyword arguments, so a level's `_rasterize_kwargs` fit."""
        cabins = cabins if cabins is not None else shapely.MultiPolygon()
        min_x, min_y, max_x, max_y = rooms.bounds
        for openings in (doors, cabins):
            if not openings.is_empty:
                dmin_x, dmin_y, dmax_x, dmax_y = openings.bounds
                min_x = min(min_x, dmin_x)
                min_y = min(min_y, dmin_y)
                max_x = max(max_x, dmax_x)
                max_y = max(max_y, dmax_y)

        width = max_x - min_x
        height = max_y - min_y
        scaling_factor = 1 / resolution
        # World metres to pixels, y flipped: the three steps compose into one matrix. A whole
        # collection goes through in one crossing into shapely, where per part the crossing costs
        # far more than the arithmetic.
        to_pixels = [
            scaling_factor,
            0.0,
            0.0,
            -scaling_factor,
            -scaling_factor * min_x,
            scaling_factor * (min_y + height),
        ]
        size = (math.ceil(width / resolution) + 2 * padding, math.ceil(height / resolution) + 2 * padding)
        return PixelFrame(to_pixels, size, padding, (min_x - padding * resolution, min_y - padding * resolution))

    @classmethod
    def generate_png(
        cls,
        rooms: shapely.MultiPolygon,
        doors: shapely.MultiPolygon,
        walls: shapely.MultiLineString,
        resolution: float = 0.01,
        padding: int = 5,
        *,
        static_objects: Iterable[tuple[str, shapely.Polygon]] = (),
        asset_color: str | None = "grey",
        asset_name_color: str | None = "blue",
        cabins: shapely.MultiPolygon | None = None,
    ) -> tuple[bytes, tuple[float, float]]:
        """
        Generate a PNG image of the map with the given elements.

        Args:
            rooms (shapely.MultiPolygon): MultiPolygon representing the rooms in the map.
            doors (shapely.MultiPolygon): MultiPolygon representing the doors in the map.
            walls (shapely.MultiLineString): MultiLineString representing the walls in the map.
            resolution (float): Size of each pixel in meters.
            padding (int): Number of pixels to pad around the map.
            show_obj_name (bool): Whether to display object names on the map.
            static_objects (Optional[List[Tuple[str, shapely.Polygon]]]): Optional list of (name, Polygon) tuples for static objects to draw.
            asset_color (str | None): Color used to fill static objects.
            asset_name_color (str | None): Color used for static object names.
            cabins (Optional[shapely.MultiPolygon]): Elevator cabins and doorways, drawn free beneath the walls.
        """
        img, origin = cls._render_image(
            rooms,
            doors,
            walls,
            resolution=resolution,
            padding=padding,
            static_objects=static_objects,
            asset_color=asset_color,
            asset_name_color=asset_name_color,
            cabins=cabins,
        )

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue(), origin

    @classmethod
    def rasterize(
        cls,
        rooms: shapely.MultiPolygon,
        doors: shapely.MultiPolygon,
        walls: shapely.MultiLineString,
        resolution: float = 0.01,
        padding: int = 5,
        *,
        static_objects: Iterable[tuple[str, shapely.Polygon]] = (),
        asset_color: str | None = "grey",
        asset_name_color: str | None = "blue",
        cabins: shapely.MultiPolygon | None = None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Like `generate_png` but returns a uint8 grayscale array (255=free, 0=occupied)."""
        img, origin = cls._render_image(
            rooms,
            doors,
            walls,
            resolution=resolution,
            padding=padding,
            static_objects=static_objects,
            asset_color=asset_color,
            asset_name_color=asset_name_color,
            cabins=cabins,
        )
        return np.asarray(img.convert('L'), dtype=np.uint8), origin

    @classmethod
    def generate_map_yaml(cls, resolution: float, filename: str, origin: tuple[float, float]) -> str:
        return typing.cast(
            str,
            yaml.safe_dump(
                {
                    'free_thresh': 0.1,
                    'image': filename,
                    'negate': 0,
                    'occupied_thresh': 0.9,
                    'origin': [*origin, 0],
                    'resolution': resolution,
                }
            ),
        )
