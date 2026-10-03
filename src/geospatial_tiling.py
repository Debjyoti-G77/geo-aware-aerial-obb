"""Deterministic raster tiling and geospatial OBB coordinate transforms.

Pixel-coordinate convention
---------------------------
Raster pixel space is treated as a continuous coordinate system whose outer
extent is ``[0, width] x [0, height]``. Integer row/column indices refer to
pixels, while model-produced polygon vertices are continuous coordinates and
are transformed directly with the raster affine transform. Consequently, no
implicit half-pixel offset is added to model vertices.

Tiling convention
-----------------
The integer stride is ``round_half_up(tile_size * (1 - overlap))``. The last
window on each axis is anchored to the scene boundary when the regular stride
would leave a strip. Duplicate final offsets are removed. Rasters smaller than
``tile_size`` retain their real dimensions and may be padded on the bottom and
right only when a caller explicitly requests a model-sized array.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


Point = tuple[float, float]


class GeometryError(ValueError):
    """Raised when a coordinate or tile is outside its valid source region."""


@dataclass(frozen=True)
class PixelBounds:
    """Continuous pixel bounds in ``(left, top, right, bottom)`` order."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class TileWindow:
    """One deterministic source-raster window and its provenance."""

    scene_id: str
    tile_id: str
    row_off: int
    col_off: int
    width: int
    height: int
    tile_size: int
    source_width: int
    source_height: int
    row_index: int
    col_index: int

    def __post_init__(self) -> None:
        integer_values = (
            self.row_off,
            self.col_off,
            self.width,
            self.height,
            self.tile_size,
            self.source_width,
            self.source_height,
            self.row_index,
            self.col_index,
        )
        if any(not isinstance(value, int) for value in integer_values):
            raise TypeError("Tile offsets, dimensions, and indices must be integers")
        if not self.scene_id or not self.tile_id:
            raise ValueError("scene_id and tile_id must be non-empty")
        if self.tile_size <= 0 or self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Tile and source dimensions must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Valid window dimensions must be positive")
        if self.width > self.tile_size or self.height > self.tile_size:
            raise ValueError("Valid window dimensions cannot exceed tile_size")
        if self.row_off < 0 or self.col_off < 0:
            raise ValueError("Window offsets cannot be negative")
        if self.row_off + self.height > self.source_height:
            raise ValueError("Window exceeds source raster height")
        if self.col_off + self.width > self.source_width:
            raise ValueError("Window exceeds source raster width")

    @property
    def valid_region(self) -> PixelBounds:
        """Real-image region in tile-local continuous pixel coordinates."""

        return PixelBounds(0.0, 0.0, float(self.width), float(self.height))

    @property
    def padded_region(self) -> PixelBounds:
        """Full model-array region, including any bottom/right padding."""

        return PixelBounds(0.0, 0.0, float(self.tile_size), float(self.tile_size))

    @property
    def source_bounds(self) -> PixelBounds:
        """Real tile bounds in full-scene continuous pixel coordinates."""

        return PixelBounds(
            float(self.col_off),
            float(self.row_off),
            float(self.col_off + self.width),
            float(self.row_off + self.height),
        )

    @property
    def has_padding(self) -> bool:
        return self.width < self.tile_size or self.height < self.tile_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "tile_id": self.tile_id,
            "row_off": self.row_off,
            "col_off": self.col_off,
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "valid_image_region": self.valid_region.as_tuple(),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "tile_bounds_source_pixels": self.source_bounds.as_tuple(),
            "row_index": self.row_index,
            "col_index": self.col_index,
        }


@dataclass(frozen=True)
class TileRead:
    """A windowed raster read and a mask distinguishing real/padded pixels."""

    tile: TileWindow
    data: Any
    valid_mask: Any


@dataclass(frozen=True)
class TilePlanValidation:
    complete_coverage: bool
    no_duplicates: bool
    all_within_source: bool

    @property
    def valid(self) -> bool:
        return self.complete_coverage and self.no_duplicates and self.all_within_source


@dataclass(frozen=True)
class GeospatialDetection:
    """Geometry and provenance for a future geospatial OBB detection."""

    scene_id: str
    tile_id: str
    class_id: int
    class_name: str
    confidence: float
    local_obb_vertices: tuple[Point, Point, Point, Point]
    global_pixel_obb_vertices: tuple[Point, Point, Point, Point]
    geographic_polygon_vertices: tuple[Point, Point, Point, Point]
    source_crs: str
    output_crs: str
    row_off: int
    col_off: int
    tile_bounds_source_pixels: tuple[float, float, float, float]
    tile_geographic_polygon: tuple[Point, Point, Point, Point]
    tile_geographic_bounds: tuple[float, float, float, float]


def stride_from_overlap(tile_size: int = 1024, overlap: float = 0.20) -> int:
    """Return a positive integer stride using round-half-up for positive values."""

    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be finite and in the range [0, 1)")
    raw_stride = tile_size * (1.0 - overlap)
    return max(1, int(math.floor(raw_stride + 0.5)))


def _axis_offsets(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Raster dimensions must be positive integers")
    if length <= tile_size:
        return (0,)

    final_offset = length - tile_size
    offsets = list(range(0, final_offset + 1, stride))
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return tuple(dict.fromkeys(offsets))


def generate_tile_windows(
    scene_id: str,
    source_width: int,
    source_height: int,
    tile_size: int = 1024,
    overlap: float = 0.20,
) -> Iterable[TileWindow]:
    """Yield a deterministic, row-major, complete set of raster windows."""

    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    stride = stride_from_overlap(tile_size, overlap)
    col_offsets = _axis_offsets(source_width, tile_size, stride)
    row_offsets = _axis_offsets(source_height, tile_size, stride)

    for row_index, row_off in enumerate(row_offsets):
        height = min(tile_size, source_height - row_off)
        for col_index, col_off in enumerate(col_offsets):
            width = min(tile_size, source_width - col_off)
            tile_id = f"{scene_id}__r{row_off:06d}_c{col_off:06d}"
            yield TileWindow(
                scene_id=scene_id,
                tile_id=tile_id,
                row_off=row_off,
                col_off=col_off,
                width=width,
                height=height,
                tile_size=tile_size,
                source_width=source_width,
                source_height=source_height,
                row_index=row_index,
                col_index=col_index,
            )


def _intervals_cover(
    intervals: Sequence[tuple[int, int]], extent: int
) -> bool:
    if not intervals:
        return False
    cursor = 0
    for start, end in sorted(intervals):
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= extent:
            return True
    return cursor >= extent


def validate_tile_plan(
    windows: Sequence[TileWindow], source_width: int, source_height: int
) -> TilePlanValidation:
    """Validate 2-D coverage without allocating a full-scene coverage array."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive")
    if not windows:
        return TilePlanValidation(False, True, False)

    keys = [
        (window.row_off, window.col_off, window.width, window.height)
        for window in windows
    ]
    no_duplicates = len(keys) == len(set(keys))
    all_within_source = all(
        window.source_width == source_width
        and window.source_height == source_height
        and window.row_off >= 0
        and window.col_off >= 0
        and window.row_off + window.height <= source_height
        and window.col_off + window.width <= source_width
        for window in windows
    )

    y_edges = {0, source_height}
    for window in windows:
        y_edges.add(window.row_off)
        y_edges.add(window.row_off + window.height)
    sorted_y = sorted(y_edges)
    complete = all_within_source
    for top, bottom in zip(sorted_y, sorted_y[1:]):
        if bottom <= top:
            continue
        midpoint = (top + bottom) / 2.0
        horizontal_intervals = [
            (window.col_off, window.col_off + window.width)
            for window in windows
            if window.row_off <= midpoint < window.row_off + window.height
        ]
        if not _intervals_cover(horizontal_intervals, source_width):
            complete = False
            break

    return TilePlanValidation(complete, no_duplicates, all_within_source)


def _four_finite_points(points: Sequence[Sequence[float]]) -> tuple[Point, Point, Point, Point]:
    if len(points) != 4:
        raise GeometryError("An OBB must contain exactly four vertices")
    if any(len(point) != 2 for point in points):
        raise GeometryError("Every OBB vertex must contain exactly two coordinates")
    converted = tuple((float(point[0]), float(point[1])) for point in points)
    if any(not math.isfinite(value) for point in converted for value in point):
        raise GeometryError("OBB coordinates must be finite")
    return converted  # type: ignore[return-value]


def points_within_valid_tile_region(
    points: Sequence[Sequence[float]], tile: TileWindow
) -> bool:
    """Return whether continuous coordinates lie in the tile's real-image extent."""

    try:
        converted = _four_finite_points(points)
    except (GeometryError, IndexError, TypeError):
        return False
    return all(0.0 <= x <= tile.width and 0.0 <= y <= tile.height for x, y in converted)


def local_obb_to_global_pixels(
    local_vertices: Sequence[Sequence[float]],
    tile: TileWindow,
    *,
    validate: bool = True,
) -> tuple[Point, Point, Point, Point]:
    """Translate arbitrary four-corner OBB vertices into full-scene pixels."""

    local = _four_finite_points(local_vertices)
    if validate and not points_within_valid_tile_region(local, tile):
        raise GeometryError(
            "An OBB vertex lies in padding or outside the tile's valid image region"
        )
    global_vertices = tuple(
        (x + tile.col_off, y + tile.row_off) for x, y in local
    )
    if validate and any(
        not (0.0 <= col <= tile.source_width and 0.0 <= row <= tile.source_height)
        for col, row in global_vertices
    ):
        raise GeometryError("A translated OBB vertex lies outside the source raster")
    return global_vertices  # type: ignore[return-value]


def global_pixels_to_world(
    global_points: Sequence[Sequence[float]], transform: Any
) -> tuple[Point, ...]:
    """Apply an affine transform directly to continuous ``(column, row)`` points."""

    result: list[Point] = []
    for point in global_points:
        if len(point) != 2:
            raise GeometryError("Every point must contain exactly two coordinates")
        col, row = float(point[0]), float(point[1])
        if not math.isfinite(col) or not math.isfinite(row):
            raise GeometryError("Pixel coordinates must be finite")
        x, y = transform @ (col, row)
        result.append((float(x), float(y)))
    return tuple(result)


def world_to_global_pixels(
    world_points: Sequence[Sequence[float]], transform: Any
) -> tuple[Point, ...]:
    """Invert an affine transform for continuous spatial coordinates."""

    inverse = ~transform
    result: list[Point] = []
    for point in world_points:
        if len(point) != 2:
            raise GeometryError("Every point must contain exactly two coordinates")
        x, y = float(point[0]), float(point[1])
        col, row = inverse @ (x, y)
        result.append((float(col), float(row)))
    return tuple(result)


def world_points_to_crs84(
    world_points: Sequence[Sequence[float]], source_crs: Any
) -> tuple[Point, ...]:
    """Normalize spatial coordinates to CRS84 ``(longitude, latitude)`` order."""

    from pyproj import CRS, Transformer

    source = CRS.from_user_input(source_crs)
    target = CRS.from_user_input("OGC:CRS84")
    transformer = Transformer.from_crs(source, target, always_xy=True)
    result: list[Point] = []
    for point in world_points:
        if len(point) != 2:
            raise GeometryError("Every point must contain exactly two coordinates")
        longitude, latitude = transformer.transform(float(point[0]), float(point[1]))
        result.append((float(longitude), float(latitude)))
    return tuple(result)


def global_pixels_to_crs84(
    global_points: Sequence[Sequence[float]], transform: Any, source_crs: Any
) -> tuple[Point, ...]:
    return world_points_to_crs84(
        global_pixels_to_world(global_points, transform), source_crs
    )


def tile_geographic_polygon(
    tile: TileWindow, transform: Any, source_crs: Any
) -> tuple[Point, Point, Point, Point]:
    bounds = tile.source_bounds
    pixel_corners = (
        (bounds.left, bounds.top),
        (bounds.right, bounds.top),
        (bounds.right, bounds.bottom),
        (bounds.left, bounds.bottom),
    )
    return global_pixels_to_crs84(pixel_corners, transform, source_crs)  # type: ignore[return-value]


def geographic_bounds(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` from longitude/latitude points."""

    if not points:
        raise GeometryError("At least one geographic point is required")
    longitudes = [float(point[0]) for point in points]
    latitudes = [float(point[1]) for point in points]
    return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))


def build_geospatial_detection(
    *,
    tile: TileWindow,
    class_id: int,
    class_name: str,
    confidence: float,
    local_obb_vertices: Sequence[Sequence[float]],
    transform: Any,
    source_crs: Any,
) -> GeospatialDetection:
    """Build a model-independent OBB record with complete tile provenance."""

    local = _four_finite_points(local_obb_vertices)
    global_pixels = local_obb_to_global_pixels(local, tile)
    geographic = global_pixels_to_crs84(global_pixels, transform, source_crs)
    tile_geo = tile_geographic_polygon(tile, transform, source_crs)
    return GeospatialDetection(
        scene_id=tile.scene_id,
        tile_id=tile.tile_id,
        class_id=int(class_id),
        class_name=str(class_name),
        confidence=float(confidence),
        local_obb_vertices=local,
        global_pixel_obb_vertices=global_pixels,
        geographic_polygon_vertices=geographic,  # type: ignore[arg-type]
        source_crs=str(source_crs),
        output_crs="OGC:CRS84",
        row_off=tile.row_off,
        col_off=tile.col_off,
        tile_bounds_source_pixels=tile.source_bounds.as_tuple(),
        tile_geographic_polygon=tile_geo,
        tile_geographic_bounds=geographic_bounds(tile_geo),
    )


def read_tile(
    dataset: Any,
    tile: TileWindow,
    *,
    pad_to_tile_size: bool = False,
    fill_value: float | int | None = None,
) -> TileRead:
    """Read one raster window, optionally padding only its bottom/right edges."""

    import numpy as np
    from rasterio.windows import Window

    if dataset.width != tile.source_width or dataset.height != tile.source_height:
        raise ValueError("Tile source dimensions do not match the open raster")
    window = Window(tile.col_off, tile.row_off, tile.width, tile.height)
    data = dataset.read(window=window, boundless=False)
    expected_shape = (dataset.count, tile.height, tile.width)
    if data.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected window shape {data.shape}; expected {expected_shape}"
        )

    if not pad_to_tile_size:
        return TileRead(
            tile=tile,
            data=data,
            valid_mask=np.ones((tile.height, tile.width), dtype=bool),
        )

    fill = dataset.nodata if fill_value is None else fill_value
    if fill is None:
        fill = 0
    padded = np.full(
        (dataset.count, tile.tile_size, tile.tile_size), fill, dtype=data.dtype
    )
    padded[:, : tile.height, : tile.width] = data
    valid_mask = np.zeros((tile.tile_size, tile.tile_size), dtype=bool)
    valid_mask[: tile.height, : tile.width] = True
    return TileRead(tile=tile, data=padded, valid_mask=valid_mask)


def tiling_parameters_from_config(config: Mapping[str, Any]) -> tuple[int, float]:
    """Read tiling settings from the project's existing configuration mapping."""

    try:
        try:
            from .project_config import ConfigError, config_value
        except ImportError:
            from project_config import ConfigError, config_value

        tile_size = int(config_value(config, "tiling", "tile_size"))
        overlap = float(config_value(config, "tiling", "overlap"))
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Config must define numeric tiling.tile_size and tiling.overlap") from exc
    stride_from_overlap(tile_size, overlap)
    return tile_size, overlap
