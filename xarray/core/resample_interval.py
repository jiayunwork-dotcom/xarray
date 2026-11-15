"""Resampling onto an arbitrary, user supplied time grid.

The public entry point is :func:`resample_interval`, which backs
:py:meth:`xarray.Dataset.resample_interval`.  Unlike
:py:meth:`xarray.Dataset.resample`, which groups records by their time label
(and therefore cannot split a record that straddles two output cells), this
implementation works directly with the *bounds* of each source interval and
integrates the source values over the target cells.

Two aggregation semantics are supported:

``"mean"``
    The value is an average that holds over the whole source interval.  It is
    weighted by the time actually falling inside each target cell and divided
    by the covered duration, i.e. the time weighted mean over the covered
    part of the cell.

``"sum"``
    The value is a total accumulated over the source interval that is
    produced uniformly in time.  Each target cell receives the fraction of
    the total proportional to the overlapping duration.  The result is *not*
    renormalized by the target cell duration.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from xarray.core.indexes import create_default_index_implicit
from xarray.core.utils import is_duck_array
from xarray.core.variable import Variable, calculate_dimensions
from xarray.namedarray.utils import is_duck_dask_array

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset


# ``np.datetime_data`` units that convert to an exact number of nanoseconds.
# Month and year units are deliberately excluded because they have no fixed
# length.
_UNIT_NS: dict[str, int] = {
    "W": 604_800 * 10**9,
    "D": 86_400 * 10**9,
    "h": 3_600 * 10**9,
    "m": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
}

# Units in which output datetime64 values may be rendered, coarsest first.
_UNITS_COARSE_TO_FINE = ("D", "h", "m", "s", "ms", "us", "ns")

_VALID_KINDS = frozenset({"mean", "sum"})

# Largest tick magnitude that keeps every intermediate accumulation safely
# inside int64.  Coverage is therefore computed with exact integers whenever
# possible, which makes the ``coverage >= min_coverage`` comparison exact even
# when the threshold is hit on the nose.
_MAX_EXACT_TICKS = 2**61


class _TimeSystem:
    """The calendar/time system shared by all time arrays of a call."""

    __slots__ = ("calendar", "kind")

    def __init__(self, kind: str, calendar: str | None = None) -> None:
        self.calendar = calendar
        self.kind = kind

    def compatible(self, other: _TimeSystem) -> bool:
        return self.kind == other.kind and self.calendar == other.calendar

    def describe(self) -> str:
        if self.kind == "cftime":
            return f"cftime calendar {self.calendar!r}"
        return "numpy datetime64"


def _get_cftime():
    try:
        import cftime
    except ImportError:  # pragma: no cover - cftime is optional
        return None
    return cftime


def _is_cftime_scalar(value: Any, cftime: Any) -> bool:
    return cftime is not None and isinstance(value, cftime.datetime)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(np.isnat(value))
    except (TypeError, ValueError):
        return False


def _is_timezone_aware(value: Any) -> bool:
    if isinstance(value, np.datetime64):
        return False
    tzinfo = getattr(value, "tzinfo", None)
    return tzinfo is not None


def _as_time_array(values: Any) -> np.ndarray:
    """Return a numpy view without silently promoting datetime64 units.

    ``np.asarray`` promotes a list of ``np.datetime64`` scalars with mixed
    units to the finest unit, which can overflow (for example year 1200 with
    seconds promoted to nanoseconds).  Plain sequences are therefore copied
    into an ``object`` array element by element, while existing numpy arrays
    keep their unit.
    """
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, pd.Index):
        return np.asarray(values)
    if isinstance(values, list | tuple):
        array = np.empty(len(values), dtype=object)
        array[:] = list(values)
        return array
    return np.asarray(values)


def _classify_array(values: np.ndarray, what: str, dtype: Any = None) -> _TimeSystem:
    """Return the time system of ``values``, raising for unsupported input.

    ``dtype`` optionally overrides the dtype reported by ``values``; this is
    needed for timezone-aware coordinates, whose underlying values are plain
    ``datetime.datetime`` objects while the variable dtype still carries the
    timezone.
    """
    if dtype is None:
        dtype = values.dtype

    if isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"{what} has a timezone-aware dtype ({dtype!r}); timezone-aware "
            "times are not supported"
        )
    if np.issubdtype(dtype, np.datetime64):
        return _TimeSystem("numpy")
    if dtype != np.dtype("O"):
        raise ValueError(
            f"{what} must contain numpy datetime64 or cftime datetime values, "
            f"got dtype {dtype!r}"
        )

    cftime = _get_cftime()
    flat = values.ravel()
    if flat.size == 0:
        raise ValueError(f"{what} must contain at least one time value")

    calendars: set[str] = set()
    saw_numpy = False
    saw_cftime = False
    for value in flat:
        if _is_missing_scalar(value):
            raise ValueError(f"{what} contains missing time values (NaT)")
        if _is_timezone_aware(value):
            raise ValueError(
                f"{what} contains timezone-aware time values; timezone-aware "
                "times are not supported"
            )
        if isinstance(value, np.datetime64):
            saw_numpy = True
        elif _is_cftime_scalar(value, cftime):
            saw_cftime = True
            calendars.add(value.calendar)
        else:
            raise ValueError(
                f"{what} must contain numpy datetime64 or cftime datetime "
                f"values, got {type(value).__name__!r}"
            )

    if saw_numpy and saw_cftime:
        raise ValueError(f"{what} mixes numpy datetime64 and cftime datetime values")
    if saw_cftime:
        if len(calendars) != 1:
            raise ValueError(
                f"{what} mixes multiple cftime calendars: {sorted(calendars)}"
            )
        return _TimeSystem("cftime", calendars.pop())
    if saw_numpy:
        return _TimeSystem("numpy")
    raise ValueError(f"{what} contains no time values")


def _numpy_scalar_ticks(value: np.datetime64, what: str) -> int:
    unit, count = np.datetime_data(value.dtype)
    if unit not in _UNIT_NS:
        raise ValueError(
            f"{what} uses unsupported datetime64 unit {unit!r}; supported units "
            "are W, D, h, m, s, ms, us and ns"
        )
    return int(value.view("i8")) * count * _UNIT_NS[unit]


def _to_ticks(values: np.ndarray, system: _TimeSystem, what: str) -> np.ndarray:
    """Convert a time array to exact integer ticks (nominal nanoseconds)."""
    if system.kind == "numpy":
        if values.dtype == np.dtype("O"):
            out = np.empty(values.shape, dtype=object)
            flat_out = out.ravel()
            for index, value in enumerate(values.ravel()):
                if _is_missing_scalar(value):
                    raise ValueError(f"{what} contains missing time values (NaT)")
                if not isinstance(value, np.datetime64):
                    raise ValueError(
                        f"{what} must contain numpy datetime64 values, got "
                        f"{type(value).__name__!r}"
                    )
                flat_out[index] = _numpy_scalar_ticks(value, what)
            return out

        if np.isnat(values).any():
            raise ValueError(f"{what} contains missing time values (NaT)")
        unit, count = np.datetime_data(values.dtype)
        if unit not in _UNIT_NS:
            raise ValueError(
                f"{what} uses unsupported datetime64 unit {unit!r}; supported "
                "units are W, D, h, m, s, ms, us and ns"
            )
        factor = count * _UNIT_NS[unit]
        native = values.astype("int64").astype(object)
        if factor == 1:
            return native
        return native * factor

    cftime = _get_cftime()
    assert cftime is not None
    calendar = system.calendar
    assert calendar is not None
    for value in values.ravel():
        if _is_missing_scalar(value) or not _is_cftime_scalar(value, cftime):
            raise ValueError(f"{what} contains missing or invalid time values")
    numbers = cftime.date2num(
        values, "microseconds since 0001-01-01 00:00:00", calendar=calendar
    )
    # One microsecond is 1000 nominal nanoseconds.
    return np.asarray(numbers, dtype=np.int64).astype(object) * 1000


def _integer_ticks(
    arrays: list[np.ndarray],
) -> tuple[list[np.ndarray], int, int]:
    """Reduce object tick arrays to int64 relative to a common origin.

    All tick values are exact Python integers (nominal nanoseconds).  They are
    first shifted by the minimum value and then divided by the greatest common
    divisor of the shifted values.  Both steps are exact multiples and preserve
    ordering, durations and containment, so an array of year-1200
    ``datetime64[s]`` values stays at second resolution instead of overflowing
    when converted to nanoseconds.  Shifting also keeps the absolute
    magnitude irrelevant, so mixing resolutions cannot overflow int64.

    Returns the reduced arrays, the nanoseconds per reduced tick, and the
    origin in nanoseconds.
    """
    flat = [int(value) for array in arrays for value in np.ravel(array)]
    origin = min(flat)

    shifted = [value - origin for value in flat]
    divisor = 0
    for value in shifted:
        divisor = math.gcd(divisor, value)
    if divisor == 0:
        divisor = 1

    span = max(shifted) // divisor
    if span > _MAX_EXACT_TICKS:
        # Needs a span of ~73 Gyr at 1 s to trigger; kept as a guard against
        # pathological cftime ranges.  Rounding to a coarser grid keeps
        # ordering and containment exact though durations could be truncated.
        divisor = divisor * -(-span // _MAX_EXACT_TICKS)

    reduced = [
        np.asarray(
            [(int(value) - origin) // divisor for value in np.ravel(array)],
            dtype=np.int64,
        ).reshape(np.shape(array))
        for array in arrays
    ]
    return reduced, divisor, origin



def _check_target_edges(target_edges: Any) -> np.ndarray:
    array = _as_time_array(target_edges)
    if array.ndim != 1:
        raise ValueError(
            "target_edges must be a one-dimensional sequence of time edges, "
            f"got an array with {array.ndim} dimension(s)"
        )
    if array.size < 2:
        raise ValueError("target_edges must contain at least two elements")
    return array


def _resolve_bounds_name(
    dataset: Dataset, dim: Hashable, bounds: Hashable | None
) -> Hashable:
    coord = dataset._variables[dim]
    attrs_ref = coord.attrs.get("bounds")
    encoding_ref = coord.encoding.get("bounds")

    if bounds is not None:
        name = bounds
    else:
        if attrs_ref is not None and encoding_ref is not None:
            if attrs_ref != encoding_ref:
                raise ValueError(
                    f"conflicting bounds references for dimension {dim!r}: "
                    f"attrs says {attrs_ref!r} but encoding says {encoding_ref!r}"
                )
        name = attrs_ref if attrs_ref is not None else encoding_ref
        if name is None:
            raise ValueError(
                f"no bounds reference found for dimension {dim!r}; pass the name "
                "of the bounds variable explicitly"
            )

    if name not in dataset._variables:
        raise ValueError(
            f"bounds variable {name!r} referenced by dimension {dim!r} is not "
            "present in the dataset"
        )
    return name


def _validate_bounds_variable(
    dataset: Dataset, name: Hashable, dim: Hashable, length: int
) -> Variable:
    variable = dataset._variables[name]
    dims = variable.dims
    if len(dims) != 2 or dims[0] != dim or dims[1] == dim or variable.shape[1] != 2:
        raise ValueError(
            f"bounds variable {name!r} must have dimensions ({dim!r}, <other "
            f"length-2 dimension>), got {dims!r} with shape "
            f"{tuple(variable.shape)!r}"
        )
    if variable.shape[0] == 0:
        raise ValueError(f"bounds variable {name!r} must contain at least one interval")
    if variable.shape[0] != length:
        raise ValueError(
            f"bounds variable {name!r} has {variable.shape[0]} interval(s) but "
            f"coordinate {dim!r} has {length} value(s)"
        )
    return variable


def _validate_kinds(
    dataset: Dataset,
    kinds: Mapping[Hashable, str],
    dim: Hashable,
    bounds_name: Hashable,
) -> dict[Hashable, str]:
    if not isinstance(kinds, Mapping):
        raise ValueError(
            "kinds must be a mapping from variable name to 'mean' or 'sum'"
        )

    required = [
        name
        for name, variable in dataset.data_vars.items()
        if dim in variable.dims and name != bounds_name
    ]
    provided = set(kinds)

    extra = provided - set(required)
    if extra:
        raise ValueError(
            "kinds contains variable(s) that are not data variables depending on "
            f"dimension {dim!r}: {sorted(map(repr, extra))}"
        )
    missing = set(required) - provided
    if missing:
        raise ValueError(
            "kinds is missing data variable(s) depending on dimension "
            f"{dim!r}: {sorted(map(repr, missing))}"
        )

    resolved: dict[Hashable, str] = {}
    for name in required:
        kind = kinds[name]
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"kind for variable {name!r} must be 'mean' or 'sum', got {kind!r}"
            )
        dtype = dataset._variables[name].dtype
        try:
            supported = np.issubdtype(dtype, np.integer) or np.issubdtype(
                dtype, np.floating
            )
        except TypeError:
            supported = False
        if not supported:
            raise ValueError(
                f"variable {name!r} has unsupported dtype {dtype!r}; only real "
                "integer and floating point values are supported"
            )
        resolved[name] = kind
    return resolved


def _validate_min_coverage(min_coverage: Any) -> float:
    if isinstance(min_coverage, bool | np.bool_) or not isinstance(min_coverage, Real):
        raise ValueError(
            f"min_coverage must be a real number in [0, 1], got {min_coverage!r}"
        )
    value = float(min_coverage)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"min_coverage must be a finite number in [0, 1], got {min_coverage!r}"
        )
    return value


def _as_float64(values: Any) -> Any:
    """Cast to float64, keeping duck arrays (e.g. dask) lazy."""
    if not is_duck_array(values):
        values = np.asarray(values)
    return values.astype(np.float64)


def _shape_with(ndim: int, axis: int, length: int) -> tuple[int, ...]:
    shape = [1] * ndim
    shape[axis] = length
    return tuple(shape)


def _segments(
    starts: np.ndarray, ends: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition the target span into elementary segments.

    Every segment lies inside exactly one source interval (or in a gap) and
    inside exactly one target cell, so reducing the segments of one cell
    only ever combines contributions that belong to that cell.  Returns the
    segment lengths (exact integer ticks), the source interval index below
    each segment (``-1`` for gaps), the target cell index below each
    segment and the first segment index of every target cell.  Every target
    cell owns at least one segment, so grouped reductions never see an
    empty group.  The number of segments is at most
    ``2 * n_sources + n_targets``.
    """
    lo = edges[0]
    hi = edges[-1]
    clipped_starts = np.clip(starts, lo, hi)
    clipped_ends = np.clip(ends, lo, hi)
    inside = clipped_ends > clipped_starts
    points = np.concatenate([edges, clipped_starts[inside], clipped_ends[inside]])
    bounds = np.unique(points)
    left = bounds[:-1]
    length = np.diff(bounds)

    candidate = np.searchsorted(starts, left, side="right") - 1
    valid = (candidate >= 0) & (left < ends[np.clip(candidate, 0, None)])
    source = np.where(valid, candidate, np.int64(-1))

    target = np.searchsorted(edges, left, side="right") - 1
    target_starts = np.searchsorted(
        target, np.arange(edges.size - 1, dtype=target.dtype), side="left"
    )
    return (
        length,
        source.astype(np.intp),
        target.astype(np.intp),
        target_starts.astype(np.intp),
    )


class _SegmentBlocks:
    """Block layout used to reduce the segment axis lazily with dask.

    Whole groups (target cells) are packed into blocks of roughly
    ``block`` segments so that no group straddles a task boundary and the
    reduction never has to combine partial groups across tasks.  A single
    group larger than ``block`` forms its own block, which keeps the work
    proportional to the actual overlap structure.  The layout depends only
    on the (shared) segment geometry and the chunking of the source data.
    """

    __slots__ = ("group_counts", "local_starts", "seg_sizes")

    def __init__(self, starts: np.ndarray, n_segments: int, chunk_hint: int) -> None:
        counts = np.diff(np.append(starts, n_segments))
        block = max(1024, 2 * chunk_hint)
        cumulative = np.cumsum(counts)
        total = int(cumulative[-1])
        cuts = np.arange(block, total, block)
        edges = np.unique(
            np.concatenate(
                [
                    [0],
                    np.searchsorted(cumulative, cuts, side="right"),
                    [counts.size],
                ]
            )
        )
        seg_bounds = np.append(starts, n_segments)
        self.seg_sizes = tuple(int(s) for s in np.diff(seg_bounds[edges]))
        self.group_counts = tuple(int(g) for g in np.diff(edges))
        self.local_starts = [
            starts[edges[i] : edges[i + 1]] - seg_bounds[edges[i]]
            for i in range(edges.size - 1)
        ]


def _take_segments(values: Any, index: np.ndarray, axis: int, layout: Any) -> Any:
    """Gather ``values`` along ``axis`` and align dask chunks to ``layout``."""
    out = np.take(values, index, axis=axis)
    if layout is not None and is_duck_dask_array(out):
        out = out.rechunk({axis: layout.seg_sizes})
    return out


def _dask_segmented_reduce(
    values: Any, layout: _SegmentBlocks, axis: int, op: str
) -> Any:
    """Reduce segments within each group along ``axis``, block by block."""
    import dask.array as dask_array

    ufunc = np.maximum if op == "maximum" else np.add
    local_starts = layout.local_starts

    def _reduce(block: Any, block_info: Any = None) -> Any:
        location = block_info[0]["chunk-location"][axis]
        return ufunc.reduceat(block, local_starts[location], axis=axis)

    values = values.rechunk({axis: layout.seg_sizes})
    chunks = list(values.chunks)
    chunks[axis] = layout.group_counts
    return dask_array.map_blocks(_reduce, values, dtype=values.dtype, chunks=chunks)


def _segmented_reduce(
    values: Any, starts: np.ndarray, axis: int, op: str, layout: Any
) -> Any:
    """Reduce segments within each group along ``axis``.

    ``starts`` holds the first segment index of every group; group ``g``
    covers segments ``starts[g] .. starts[g + 1]`` (or the end of the
    axis).  Groups are never empty.  ``op`` is ``"add"`` or ``"maximum"``.
    """
    if is_duck_dask_array(values):
        return _dask_segmented_reduce(values, layout, axis, op)
    if op == "maximum":
        return np.maximum.reduceat(values, starts, axis=axis)
    return np.add.reduceat(values, starts, axis=axis)


def _ticks_to_datetime64(
    ticks: np.ndarray, divisor: int, origin: int = 0
) -> np.ndarray:
    """Render exact integer ticks as the coarsest exactly representable unit.

    ``ticks`` are offsets from ``origin`` and each tick is ``divisor``
    nanoseconds.  The coarsest unit that represents every value exactly is
    chosen, so year-1200 ``datetime64[s]`` input comes back as
    ``datetime64[s]`` instead of being forced into nanoseconds and overflowing.
    """
    ticks = np.asarray(ticks, dtype=np.int64)
    for unit in _UNITS_COARSE_TO_FINE:
        factor = _UNIT_NS[unit]
        common = math.gcd(divisor, factor)
        divisor_part = divisor // common
        factor_part = factor // common
        if np.any(ticks % factor_part != 0) or origin % factor != 0:
            continue
        native = (ticks // factor_part) * divisor_part + origin // factor
        return native.view(f"datetime64[{unit}]")
    # ``ns`` always divides the ticks, so this is unreachable.
    raise AssertionError("could not render ticks")  # pragma: no cover


def _build_output_coordinate(
    dataset: Dataset,
    dim: Hashable,
    left_edges: np.ndarray,
    bounds_name: Hashable | None,
) -> tuple[Variable, Any]:
    """Build the output dimension coordinate, dropping stale bounds links."""
    source = dataset._variables[dim]
    attrs = dict(source.attrs)
    encoding = dict(source.encoding)
    attrs.pop("bounds", None)
    encoding.pop("bounds", None)

    if bounds_name is not None:
        attrs["bounds"] = bounds_name
        encoding["bounds"] = bounds_name

    variable = Variable((dim,), left_edges, attrs=attrs, encoding=encoding)
    try:
        index, index_variables = create_default_index_implicit(
            variable.to_index_variable()
        )
    except (TypeError, ValueError, OverflowError):
        # Far out of range timestamps cannot be wrapped in a pandas index; a
        # plain coordinate still carries the labels.
        return variable, None
    return index_variables[dim], index


def resample_interval(
    dataset: Dataset,
    target_edges: Any,
    *,
    kinds: Mapping[Hashable, str],
    dim: Hashable = "time",
    bounds: Hashable | None = None,
    min_coverage: float = 1.0,
) -> tuple[Dataset, Dataset]:
    """Implementation of :py:meth:`xarray.Dataset.resample_interval`."""
    from xarray.core.dataset import Dataset

    if dim not in dataset._variables:
        raise ValueError(
            f"dimension {dim!r} must be a one-dimensional time coordinate, but no "
            "such coordinate is present"
        )
    time_variable = dataset._variables[dim]
    if time_variable.dims != (dim,):
        raise ValueError(
            f"dimension {dim!r} must be a one-dimensional time coordinate, got "
            f"dims {time_variable.dims!r}"
        )
    if time_variable.size == 0:
        raise ValueError(f"dimension {dim!r} must contain at least one time value")

    bounds_name = _resolve_bounds_name(dataset, dim, bounds)
    bounds_variable = _validate_bounds_variable(
        dataset, bounds_name, dim, time_variable.size
    )

    kinds = _validate_kinds(dataset, kinds, dim, bounds_name)
    min_coverage = _validate_min_coverage(min_coverage)

    edges_array = _check_target_edges(target_edges)

    # --- classify and convert the three time arrays ---------------------
    coord_values = np.asarray(time_variable.values)
    bounds_values = np.asarray(bounds_variable.values)
    coords_system = _classify_array(
        coord_values, f"coordinate {dim!r}", dtype=time_variable.dtype
    )
    bounds_system = _classify_array(
        bounds_values, f"bounds variable {bounds_name!r}", dtype=bounds_variable.dtype
    )
    edges_system = _classify_array(edges_array, "target_edges")

    expected = coords_system.describe()
    for what, system in (
        (f"bounds variable {bounds_name!r}", bounds_system),
        ("target_edges", edges_system),
    ):
        if not coords_system.compatible(system):
            raise ValueError(
                f"{what} uses {system.describe()} but coordinate {dim!r} uses "
                f"{expected}; all time arrays must use the same calendar"
            )

    coord_ticks = _to_ticks(coord_values, coords_system, f"coordinate {dim!r}")
    bounds_ticks = _to_ticks(
        bounds_values, bounds_system, f"bounds variable {bounds_name!r}"
    )
    edge_ticks = _to_ticks(edges_array, edges_system, "target_edges")

    (coord_int, bounds_int, edge_int), divisor, origin = _integer_ticks(
        [coord_ticks, bounds_ticks, edge_ticks]
    )

    starts = bounds_int[:, 0]
    ends = bounds_int[:, 1]
    durations = ends - starts

    # --- validate the source intervals ----------------------------------
    if durations.size == 0 or int(durations.min()) <= 0:
        raise ValueError(
            f"every source interval in {bounds_name!r} must have a positive duration"
        )

    previous_start: int | None = None
    previous_end: int | None = None
    for position in range(starts.size):
        start = int(starts[position])
        end = int(ends[position])
        if previous_start is not None and start <= previous_start:
            raise ValueError(
                "source intervals must be sorted by ascending start time; "
                f"interval {position} starts at or before interval {position - 1}"
            )
        if previous_end is not None and start < previous_end:
            raise ValueError(
                "source intervals must not overlap; interval "
                f"{position} starts before interval {position - 1} ends"
            )
        label = int(coord_int[position])
        if not (start <= label < end):
            raise ValueError(
                f"time label {coord_values[position]!r} does not fall inside its "
                f"own left-closed right-open interval "
                f"[{bounds_values[position, 0]!r}, {bounds_values[position, 1]!r})"
            )
        previous_start = start
        previous_end = end

    for position in range(edge_int.size - 1):
        if int(edge_int[position]) >= int(edge_int[position + 1]):
            raise ValueError(
                "target_edges must be strictly increasing; edge "
                f"{position + 1} is not greater than edge {position}"
            )

    # --- geometry of target cells relative to source intervals ----------
    cell_count = edge_int.size - 1
    target_ticks = edge_int[1:] - edge_int[:-1]

    # Split the target span into elementary segments.  Every segment lies
    # inside exactly one source interval (or a gap) and one target cell, so
    # a target cell is only ever influenced by source intervals it actually
    # intersects: contributions are reduced per cell instead of being
    # recovered by differencing a running total, which would let large
    # values in unrelated cells cancel small local ones (and overflow).
    seg_length, seg_source, seg_target, target_starts = _segments(
        starts, ends, edge_int
    )
    seg_count = seg_length.size
    seg_source_safe = np.clip(seg_source, 0, None)
    # Exact ticks of valid source covered by each segment (0 for gaps) and,
    # for ``sum``, the fraction of its source interval each segment
    # represents (at most 1, so terms never exceed the source magnitude).
    seg_valid_length = np.where(seg_source >= 0, seg_length, np.int64(0))
    seg_fraction = np.where(
        seg_source >= 0, seg_length / durations[seg_source_safe], 0.0
    )

    # --- numeric core, one variable at a time ---------------------------
    results: dict[Hashable, Variable] = {}
    coverages: dict[Hashable, Variable] = {}
    used_dims: set[Hashable] = {dim}

    for name, kind in kinds.items():
        source_variable = dataset._variables[name]
        used_dims.update(source_variable.dims)
        axis = source_variable.dims.index(dim)
        broadcast = _shape_with(source_variable.ndim, axis, seg_count)

        values = _as_float64(source_variable.data)
        finite = np.isfinite(values)
        filled = np.where(finite, values, np.float64(0.0))

        if is_duck_dask_array(values):
            known_chunks = [
                int(c) for c in values.chunks[axis] if not math.isnan(c)
            ]
            layout: _SegmentBlocks | None = _SegmentBlocks(
                target_starts, seg_count, max(known_chunks, default=1024)
            )
        else:
            layout = None

        # Coverage: exact integer ticks of finite data in each cell.  No
        # rounding is involved, so the coverage threshold comparison is
        # exact even when the threshold is hit on the nose.
        mask = finite.astype(np.int64)
        mask_seg = _take_segments(mask, seg_source_safe, axis, layout)
        covered_seg = mask_seg * seg_valid_length.reshape(broadcast)
        covered_ticks = _segmented_reduce(
            covered_seg, target_starts, axis, "add", layout
        )

        filled_seg = _take_segments(filled, seg_source_safe, axis, layout)

        if kind == "mean":
            # Convex combination over the cell: each weight is the share of
            # the covered duration contributed by its segment, so weights
            # are at most 1 and every partial sum stays bounded by the
            # largest magnitude in the cell.  The mean of finite values
            # therefore cannot overflow to inf.
            safe_covered = np.where(
                covered_ticks > 0, covered_ticks, np.int64(1)
            ).astype(np.float64)
            normalizer = _take_segments(safe_covered, seg_target, axis, layout)
            weights = (
                seg_valid_length.astype(np.float64).reshape(broadcast) / normalizer
            )
            integrated = _segmented_reduce(
                filled_seg * weights, target_starts, axis, "add", layout
            )
        else:
            terms = filled_seg * seg_fraction.reshape(broadcast)
            # Scale every cell by its largest term before accumulating, so
            # intermediate sums stay within the range of the true result:
            # a representable sum of finite values is never reported as inf
            # or NaN because of intermediate overflow.
            scale = _segmented_reduce(
                np.abs(terms), target_starts, axis, "maximum", layout
            )
            safe_scale = np.where(scale > 0, scale, np.float64(1.0))
            normalizer = _take_segments(safe_scale, seg_target, axis, layout)
            summed = _segmented_reduce(
                terms / normalizer, target_starts, axis, "add", layout
            )
            integrated = summed * scale

        ratio = covered_ticks.astype(np.float64) / target_ticks.astype(
            np.float64
        ).reshape(_shape_with(source_variable.ndim, axis, cell_count))
        ratio = np.clip(ratio, 0.0, 1.0)

        keep = (covered_ticks > 0) & (ratio >= min_coverage)
        result_data = np.where(keep, integrated, np.float64(np.nan))

        results[name] = Variable(
            source_variable.dims,
            result_data,
            attrs=dict(source_variable.attrs),
        )
        coverages[name] = Variable(source_variable.dims, ratio)

    # --- assemble the result dataset ------------------------------------
    if coords_system.kind == "numpy":
        left_edges = _ticks_to_datetime64(edge_int[:-1], divisor, origin)
        right_edges = _ticks_to_datetime64(edge_int[1:], divisor, origin)
    else:
        left_edges = edges_array[:-1]
        right_edges = edges_array[1:]
    out_bounds = np.stack([left_edges, right_edges], axis=1)

    dim_coordinate, dim_index = _build_output_coordinate(
        dataset, dim, left_edges, bounds_name
    )
    bounds_output = Variable(
        bounds_variable.dims,
        out_bounds,
        attrs=dict(bounds_variable.attrs),
        encoding=dict(bounds_variable.encoding),
    )

    variables: dict[Hashable, Variable] = {}
    coord_names: set[Hashable] = set()
    for name, variable in dataset._variables.items():
        if name == dim:
            variables[dim] = dim_coordinate
            coord_names.add(dim)
        elif name == bounds_name:
            variables[name] = bounds_output
            if name in dataset._coord_names:
                coord_names.add(name)
        elif name in results:
            variables[name] = results[name]
        elif dim in variable.dims:
            # auxiliary coordinates that depend on ``dim`` are dropped
            continue
        else:
            variables[name] = variable.copy(deep=False)
            if name in dataset._coord_names:
                coord_names.add(name)

    indexes: dict[Hashable, Any] = {
        name: index
        for name, index in dataset._indexes.items()
        if name in variables and name != dim
    }
    if dim_index is not None:
        indexes[dim] = dim_index

    result = Dataset._construct_direct(
        variables,
        coord_names,
        dims=calculate_dimensions(variables),
        attrs=dict(dataset.attrs),
        indexes=indexes,
        encoding=dict(dataset.encoding),
    )

    # --- assemble the coverage dataset ----------------------------------
    coverage_coordinate, coverage_index = _build_output_coordinate(
        dataset, dim, left_edges, None
    )
    coverage_variables: dict[Hashable, Variable] = {dim: coverage_coordinate}
    coverage_coord_names: set[Hashable] = {dim}
    for name, variable in dataset._variables.items():
        if name in (dim, bounds_name) or name in results:
            continue
        if name in dataset._coord_names and dim not in variable.dims:
            if set(variable.dims) <= used_dims:
                coverage_variables[name] = variable.copy(deep=False)
                coverage_coord_names.add(name)
    coverage_variables.update(coverages)

    coverage_indexes: dict[Hashable, Any] = {
        name: index
        for name, index in dataset._indexes.items()
        if name in coverage_variables and name != dim
    }
    if coverage_index is not None:
        coverage_indexes[dim] = coverage_index

    coverage = Dataset._construct_direct(
        coverage_variables,
        coverage_coord_names,
        dims=calculate_dimensions(coverage_variables),
        attrs={},
        indexes=coverage_indexes,
    )

    return result, coverage
