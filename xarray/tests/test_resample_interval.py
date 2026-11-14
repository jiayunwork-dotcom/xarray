"""Tests for :py:meth:`xarray.Dataset.resample_interval`."""

from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray.tests import assert_identical, requires_cftime, requires_dask

try:
    import cftime
except ImportError:
    cftime = None


D = "datetime64[ns]"


def _bounds_dataset(values, bounds, times=None, ref="tb", dim="time", **extra):
    """Build a Dataset with a bounds variable referenced from the coordinate."""
    values = np.asarray(values, dtype=float)
    if times is None:
        times = bounds[:, 0]
    attrs = {"bounds": ref} if ref else {}
    coords = {
        dim: (dim, np.asarray(times), attrs),
        ref: ((dim, "nv"), np.asarray(bounds)) if ref else None,
    }
    if ref:
        data = {"v": (dim, values), ref: ((dim, "nv"), np.asarray(bounds))}
        coords.pop(ref)
    else:
        data = {"v": (dim, values)}
    ds = xr.Dataset(data, coords={k: v for k, v in coords.items() if v is not None})
    if extra:
        ds = ds.assign(**extra)
    return ds


class TestResampleIntervalBasics:
    def test_unequal_duration_mean(self):
        """A 6h and an 18h mean must be weighted by their durations."""
        time = np.array(["2000-01-01T00", "2000-01-01T06"], dtype="datetime64[s]")
        bounds = np.array(
            [
                ["2000-01-01T00", "2000-01-01T06"],
                ["2000-01-01T06", "2000-01-02T00"],
            ],
            dtype="datetime64[s]",
        )
        ds = _bounds_dataset([10.0, 30.0], bounds, times=time)
        edges = np.array(["2000-01-01T00", "2000-01-02T00"], dtype="datetime64[s]")

        result, coverage = ds.resample_interval(edges, kinds={"v": "mean"})
        np.testing.assert_allclose(result["v"].values, [25.0])
        np.testing.assert_allclose(coverage["v"].values, [1.0])

        result_sum, _ = ds.resample_interval(edges, kinds={"v": "sum"})
        np.testing.assert_allclose(result_sum["v"].values, [40.0])

    def test_straddling_record_is_split(self):
        """A record crossing a target edge contributes to both cells."""
        time = np.array(["2000-01-01", "2000-01-03"], dtype=D)
        bounds = np.array(
            [
                ["2000-01-01", "2000-01-02"],
                ["2000-01-02", "2000-01-04"],
            ],
            dtype=D,
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=time)
        edges = np.array(["2000-01-01", "2000-01-03", "2000-01-05"], dtype=D)

        result, coverage = ds.resample_interval(edges, kinds={"v": "sum"})
        # Cell 1 is fully covered: 1 (whole record) + 1 (half of the 2-day
        # record).  Cell 2 only reaches 0.5 coverage, so with the default
        # ``min_coverage=1`` its value is masked but coverage is reported.
        np.testing.assert_allclose(result["v"].values, [2.0, np.nan])
        np.testing.assert_allclose(coverage["v"].values, [1.0, 0.5])

        unmasked, _ = ds.resample_interval(edges, kinds={"v": "sum"}, min_coverage=0.0)
        np.testing.assert_allclose(unmasked["v"].values, [2.0, 1.0])

    def test_spanning_many_cells(self):
        starts = np.datetime64("2000-01-01T00", "s") + np.arange(4) * np.timedelta64(
            1, "h"
        )
        ends = starts + np.timedelta64(1, "h")
        bounds = np.stack([starts, ends], axis=1)
        ds = _bounds_dataset([1.0, 2.0, 3.0, 4.0], bounds, times=starts)
        edges = np.datetime64("2000-01-01T00", "s") + np.arange(0, 5) * np.timedelta64(
            1, "h"
        )
        result, coverage = ds.resample_interval(edges, kinds={"v": "sum"})
        np.testing.assert_allclose(result["v"].values, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(coverage["v"].values, [1.0, 1.0, 1.0, 1.0])

    def test_mean_of_multiple_source_intervals(self):
        """The mean over a cell is the time weighted mean of its records."""
        t0 = np.datetime64("2000-01-01T00", "h")
        starts = t0 + np.array([0, 2, 5], dtype="timedelta64[h]")
        ends = t0 + np.array([2, 5, 10], dtype="timedelta64[h]")
        bounds = np.stack([starts, ends], axis=1)
        ds = _bounds_dataset([10.0, 20.0, 40.0], bounds, times=starts)
        edges = t0 + np.array([0, 9, 10], dtype="timedelta64[h]")
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "mean"}, min_coverage=0.0
        )
        # Cell 1: [0, 9); covers all of [0,2) (10), [2,5) (20) and four of
        # the five hours of [5,10) (40).  Weighted mean =
        # (2*10 + 3*20 + 4*40) / 9.
        np.testing.assert_allclose(
            result["v"].values, [(2 * 10 + 3 * 20 + 4 * 40) / 9, 40.0]
        )
        np.testing.assert_allclose(coverage["v"].values, [1.0, 1.0])

    def test_sum_is_not_renormalized(self):
        """A sum must not be divided by the target cell length."""
        starts = np.array(["2000-01-01T00", "2000-01-02T00"], dtype="datetime64[s]")
        bounds = np.array(
            [
                ["2000-01-01T00", "2000-01-01T06"],
                ["2000-01-02T00", "2000-01-02T06"],
            ],
            dtype="datetime64[s]",
        )
        ds = _bounds_dataset([5.0, 7.0], bounds, times=starts)
        edges = np.array(
            ["2000-01-01T00", "2000-01-01T06", "2000-01-02T06"],
            dtype="datetime64[s]",
        )
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        # The sum is not divided by the (longer) cell duration.
        np.testing.assert_allclose(result["v"].values, [5.0, 7.0])
        # Cell 2 is 24h long and only 6h are covered.
        np.testing.assert_allclose(coverage["v"].values, [1.0, 0.25])

    def test_oriented_along_other_axes(self):
        """dim need not be the first axis and extra dims are supported."""
        n = 4
        t = np.datetime64("2000-01-01", "s") + np.arange(n).astype("timedelta64[s]")
        bounds = np.stack([t, t + np.timedelta64(1, "s")], axis=1)
        values = np.arange(2 * n * 3, dtype=float).reshape(2, n, 3)
        ds = xr.Dataset(
            {
                "a": (("x", "time", "y"), values),
                "b": ("time", np.arange(n, dtype=float)),
            },
            coords={
                "time": ("time", t, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
                "x": [10, 20],
                "y": ["a", "b", "c"],
            },
        )
        edges = np.datetime64("2000-01-01", "s") + np.array(
            [0, 2, 4], dtype="timedelta64[s]"
        )
        result, coverage = ds.resample_interval(
            edges, kinds={"a": "mean", "b": "sum"}, min_coverage=0.0
        )
        assert result["a"].dims == ("x", "time", "y")
        assert result["b"].dims == ("time",)
        assert coverage["a"].dims == ("x", "time", "y")
        assert coverage["a"].shape == result["a"].shape
        np.testing.assert_allclose(result["b"].values, [1.0, 5.0])

    def test_target_can_extend_and_trim(self):
        t = np.array(["2000-01-02", "2000-01-03"], dtype=D)
        bounds = np.array(
            [["2000-01-02", "2000-01-03"], ["2000-01-03", "2000-01-04"]], dtype=D
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=t)
        # Edges extend past both sides and cut through records.
        edges = np.array(["2000-01-01", "2000-01-02T12", "2000-01-05"], dtype=D)
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        # Cell 1 [Jan1, Jan2 12) overlaps half a day of the record => 0.5.
        # Cell 2 [Jan2 12, Jan5) sees the other half day plus the whole
        # second record => 0.5 + 1 = 1.5 of 2.5 days.
        np.testing.assert_allclose(result["v"].values, [0.5, 2.5])
        np.testing.assert_allclose(coverage["v"].values, [0.5 / 1.5, 1.5 / 2.5])

    def test_dim_coordinate_and_bounds_content(self):
        t = np.array(["2000-01-01", "2000-01-03"], dtype=D)
        bounds = np.array(
            [["2000-01-01", "2000-01-02"], ["2000-01-03", "2000-01-05"]], dtype=D
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=t)
        edges = np.array(["2000-01-01", "2000-01-02", "2000-01-06"], dtype=D)
        result, _ = ds.resample_interval(edges, kinds={"v": "sum"})
        expected_edges = edges[:-1]
        assert result["time"].dims == ("time",)
        np.testing.assert_allclose(
            result["time"].values.astype("datetime64[ns]").astype("int64"),
            expected_edges.astype("int64"),
        )
        assert result["tb"].dims == ("time", "nv")
        np.testing.assert_allclose(
            result["tb"].values.astype("datetime64[ns]").astype("int64"),
            np.stack([edges[:-1], edges[1:]], axis=1).astype("int64"),
        )
        assert result["time"].attrs["bounds"] == "tb"
        assert result["time"].encoding["bounds"] == "tb"


class TestCoverage:
    def test_gap_is_not_coverage(self):
        starts = np.array(["2000-01-01", "2000-01-05"], dtype=D)
        bounds = np.array(
            [["2000-01-01", "2000-01-02"], ["2000-01-05", "2000-01-06"]], dtype=D
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=starts)
        edges = np.array(["2000-01-01", "2000-01-03", "2000-01-06"], dtype=D)
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        # Cell [Jan1, Jan3) covers only Jan1--Jan2 => 0.5; the gap is not
        # counted as valid coverage.  Cell [Jan3, Jan6) covers only
        # Jan5--Jan6 => 1/3.
        np.testing.assert_allclose(coverage["v"].values, [0.5, 1.0 / 3.0])
        np.testing.assert_allclose(result["v"].values, [1.0, 2.0])

    def test_per_position_missingness(self):
        """Coverage is per variable and per non-time position."""
        n = 3
        t = np.datetime64("2000-01-01", "s") + np.arange(n).astype("timedelta64[s]")
        bounds = np.stack([t, t + np.timedelta64(1, "s")], axis=1)
        values = np.array(
            [
                [1.0, np.nan],
                [2.0, np.inf],
                [3.0, -np.inf],
            ]
        )
        ds = xr.Dataset(
            {"v": (("time", "y"), values)},
            coords={
                "time": ("time", t, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
                "y": ["a", "b"],
            },
        )
        edges = np.datetime64("2000-01-01", "s") + np.array(
            [0, 3], dtype="timedelta64[s]"
        )
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        np.testing.assert_allclose(coverage["v"].values[0], [1.0, 0.0])
        # "a" has all finite data, "b" has none -> NaN even at min_coverage 0.
        np.testing.assert_allclose(result["v"].values[0, 0], 6.0)
        assert np.isnan(result["v"].values[0, 1])

    def test_min_coverage_threshold_boundary(self):
        """A threshold exactly equal to coverage keeps the value."""
        t = np.array(["2000-01-01", "2000-01-04"], dtype="datetime64[s]")
        bounds = np.array(
            [["2000-01-01", "2000-01-02"], ["2000-01-04", "2000-01-05"]],
            dtype="datetime64[s]",
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=t)
        edges = np.array(["2000-01-02", "2000-01-05"], dtype="datetime64[s]")
        # Cell is 3 days long and only 1 day is covered => exactly 1/3.
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=1.0 / 3.0
        )
        np.testing.assert_allclose(coverage["v"].values, [1.0 / 3.0])
        np.testing.assert_allclose(result["v"].values, [2.0])
        # A hair above the exact fraction masks the result but keeps coverage.
        result_hi, coverage_hi = ds.resample_interval(
            edges,
            kinds={"v": "sum"},
            min_coverage=float(np.nextafter(1.0 / 3.0, 1.0)),
        )
        np.testing.assert_allclose(coverage_hi["v"].values, [1.0 / 3.0])
        assert np.isnan(result_hi["v"].values[0])

    def test_no_coverage_is_nan_even_at_zero(self):
        t = np.array(["2000-01-10"], dtype="datetime64[s]")
        bounds = np.array([["2000-01-10", "2000-01-11"]], dtype="datetime64[s]")
        ds = _bounds_dataset([5.0], bounds, times=t)
        edges = np.array(["2000-01-01", "2000-01-05"], dtype="datetime64[s]")
        for kind in ("mean", "sum"):
            result, coverage = ds.resample_interval(
                edges, kinds={"v": kind}, min_coverage=0.0
            )
            assert np.isnan(result["v"].values[0])
            np.testing.assert_allclose(coverage["v"].values, [0.0])

    def test_coverage_is_dimension_less(self):
        n = 4
        t = np.datetime64("2000-01-01", "s") + np.arange(n).astype("timedelta64[s]")
        bounds = np.stack([t, t + np.timedelta64(1, "s")], axis=1)
        ds = xr.Dataset(
            {"v": (("x", "time", "y"), np.ones((2, n, 3), dtype=float))},
            coords={
                "time": ("time", t, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
                "x": [0, 1],
                "y": ["a", "b", "c"],
            },
        )
        edges = np.datetime64("2000-01-01", "s") + np.array(
            [0, 2, 4], dtype="timedelta64[s]"
        )
        result, coverage = ds.resample_interval(edges, kinds={"v": "mean"})
        assert coverage["v"].values.min() >= 0.0
        assert coverage["v"].values.max() <= 1.0
        assert coverage["v"].attrs == {}
        assert coverage["v"].dims == result["v"].dims


class TestMetadata:
    def test_preserves_static_variables_and_attrs(self):
        t = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype=D)
        bounds = np.array(
            [
                ["2000-01-01", "2000-01-02"],
                ["2000-01-02", "2000-01-03"],
                ["2000-01-03", "2000-01-04"],
            ],
            dtype=D,
        )
        ds = xr.Dataset(
            {
                "v": (("x", "time"), np.arange(6.0).reshape(2, 3), {"units": "K"}),
                "static": ("x", [1.0, 2.0], {"units": "m"}),
            },
            coords={
                "time": ("time", t, {"bounds": "tb", "axis": "T"}),
                "tb": (("time", "nv"), bounds, {"long_name": "bounds"}),
                "x": ("x", [10, 20], {"axis": "X"}),
                "aux": (("x", "time"), np.zeros((2, 3))),
            },
        )
        ds.attrs["title"] = "demo"
        edges = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype=D)
        result, coverage = ds.resample_interval(edges, kinds={"v": "mean"})

        assert result["v"].attrs == {"units": "K"}
        assert result.attrs == {"title": "demo"}
        assert_identical(result["static"], ds["static"])
        assert_identical(result["x"], ds["x"])
        assert "aux" not in result
        assert "aux" not in coverage
        assert "static" not in coverage

        # bounds reference is updated in both attrs and encoding
        assert result["time"].attrs["bounds"] == "tb"
        assert result["tb"].attrs == {"long_name": "bounds"}
        # coverage keeps no bounds reference at all
        assert "bounds" not in coverage["time"].attrs
        assert "bounds" not in coverage["time"].encoding

    def test_input_not_mutated(self):
        t = np.array(["2000-01-01", "2000-01-02"], dtype=D)
        bounds = np.array(
            [["2000-01-01", "2000-01-02"], ["2000-01-02", "2000-01-03"]], dtype=D
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=t)
        before_t = ds["time"].values.copy()
        before_b = ds["tb"].values.copy()
        before_attrs = dict(ds["time"].attrs)
        edges = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype=D)
        ds.resample_interval(edges, kinds={"v": "mean"})
        assert np.array_equal(ds["time"].values, before_t)
        assert np.array_equal(ds["tb"].values, before_b)
        assert ds["time"].attrs == before_attrs
        assert "bounds" not in ds["time"].encoding

    def test_existing_resample_still_works(self):
        times = xr.date_range("2000-01-01", periods=10, freq="h")
        ds = xr.Dataset({"a": ("time", np.arange(10.0))}, coords={"time": times})
        np.testing.assert_allclose(
            ds.resample(time="6h").mean()["a"].values, [2.5, 7.5]
        )


class TestValidation:
    def _ds(self, **kwargs):
        times = kwargs.pop(
            "times",
            np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype=D),
        )
        bounds = kwargs.pop(
            "bounds",
            np.array(
                [
                    ["2000-01-01", "2000-01-02"],
                    ["2000-01-02", "2000-01-03"],
                    ["2000-01-03", "2000-01-04"],
                ],
                dtype=D,
            ),
        )
        return _bounds_dataset([1.0, 2.0, 3.0], bounds, times=times, **kwargs)

    @pytest.mark.parametrize(
        "edges",
        [
            np.array(["2000-01-01"], dtype=D),
            np.array(["2000-01-01", "2000-01-01"], dtype=D),
            np.array(["2000-01-01", "2000-01-03", "2000-01-02"], dtype=D),
            np.array([["2000-01-01", "2000-01-02"]], dtype=D),
        ],
    )
    def test_bad_target_edges(self, edges):
        with pytest.raises(ValueError):
            self._ds().resample_interval(edges, kinds={"v": "sum"})

    def test_non_increasing_source_bounds(self):
        bounds = np.array(
            [
                ["2000-01-02", "2000-01-01"],
                ["2000-01-02", "2000-01-03"],
                ["2000-01-03", "2000-01-04"],
            ],
            dtype=D,
        )
        with pytest.raises(ValueError, match="positive duration"):
            self._ds(bounds=bounds).resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_overlapping_source_intervals(self):
        bounds = np.array(
            [
                ["2000-01-01", "2000-01-03"],
                ["2000-01-02", "2000-01-04"],
                ["2000-01-03", "2000-01-05"],
            ],
            dtype=D,
        )
        with pytest.raises(ValueError, match="overlap"):
            self._ds(bounds=bounds).resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_unsorted_source_intervals(self):
        times = np.array(["2000-01-02", "2000-01-01", "2000-01-03"], dtype=D)
        bounds = np.array(
            [
                ["2000-01-02", "2000-01-03"],
                ["2000-01-01", "2000-01-02"],
                ["2000-01-03", "2000-01-04"],
            ],
            dtype=D,
        )
        with pytest.raises(ValueError, match="ascending"):
            self._ds(times=times, bounds=bounds).resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_label_outside_own_interval(self):
        times = np.array(["2000-01-02", "2000-01-02", "2000-01-03"], dtype=D)
        with pytest.raises(ValueError, match="does not fall inside"):
            self._ds(times=times).resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_bounds_dimension_shape(self):
        ds = xr.Dataset(
            {"v": ("time", [1.0, 2.0, 3.0])},
            coords={
                "time": (
                    "time",
                    np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype=D),
                    {"bounds": "tb"},
                ),
                "tb": (
                    ("nv", "time"),
                    np.zeros((2, 3)),
                ),
            },
        )
        with pytest.raises(ValueError, match="must have dimensions"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_bounds_length_not_two(self):
        ds = xr.Dataset(
            {"v": ("time", [1.0, 2.0, 3.0])},
            coords={
                "time": (
                    "time",
                    np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype=D),
                    {"bounds": "tb"},
                ),
                "tb": (
                    ("time", "nv"),
                    np.zeros((3, 4)),
                ),
            },
        )
        with pytest.raises(ValueError, match="must have dimensions"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_missing_bounds_reference(self):
        with pytest.raises(ValueError, match="no bounds reference"):
            self._ds(ref=None).resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_conflicting_bounds_references(self):
        ds = self._ds()
        ds["time"].encoding["bounds"] = "other"
        with pytest.raises(ValueError, match="conflicting bounds"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_bounds_variable_absent(self):
        ds = self._ds()
        ds["time"].attrs["bounds"] = "nope"
        with pytest.raises(ValueError, match="not present"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_kinds_must_cover_everything(self):
        ds = self._ds()
        ds["other"] = ("time", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="missing"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )
        with pytest.raises(ValueError, match="not data variables"):
            self._ds().resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D),
                kinds={"v": "sum", "nope": "sum"},
            )

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="must be 'mean' or 'sum'"):
            self._ds().resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "max"}
            )

    @pytest.mark.parametrize(
        "dtype_values",
        [
            np.array([True, False, True]),
            np.array([1 + 1j, 2 + 0j, 3 + 0j]),
            np.array(["a", "b", "c"]),
        ],
    )
    def test_unsupported_dtypes(self, dtype_values):
        ds = self._ds()
        ds["v"] = ("time", dtype_values)
        with pytest.raises(ValueError, match="unsupported dtype"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    @pytest.mark.parametrize("min_coverage", [True, 1.5, -0.1, float("nan"), "1"])
    def test_invalid_min_coverage(self, min_coverage):
        with pytest.raises(ValueError, match="min_coverage"):
            self._ds().resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D),
                kinds={"v": "sum"},
                min_coverage=min_coverage,
            )

    def test_nat_in_coordinate(self):
        times = np.array(["2000-01-01", "NaT", "2000-01-03"], dtype=D)
        with pytest.raises(ValueError, match="NaT"):
            self._ds(times=times).resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    def test_nat_in_target_edges(self):
        with pytest.raises(ValueError, match="NaT"):
            self._ds().resample_interval(
                np.array(["2000-01-01", "NaT", "2000-01-05"], dtype=D),
                kinds={"v": "sum"},
            )

    def test_timezone_aware_rejected(self):
        times = xr.date_range("2000-01-01", periods=3, freq="D", tz="UTC")
        bounds = np.array(
            [
                ["2000-01-01", "2000-01-02"],
                ["2000-01-02", "2000-01-03"],
                ["2000-01-03", "2000-01-04"],
            ],
            dtype=D,
        )
        ds = xr.Dataset(
            {"v": ("time", [1.0, 2.0, 3.0])},
            coords={"time": times, "tb": (("time", "nv"), bounds)},
        )
        ds["time"].attrs["bounds"] = "tb"
        with pytest.raises(ValueError, match="timezone"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )


class TestCalendarHandling:
    def test_numpy_second_resolution_out_of_ns_range(self):
        """Year 1200 at second resolution must not overflow to ns."""
        times = np.array(["1200-01-01", "1200-01-05"], dtype="datetime64[s]")
        bounds = np.array(
            [
                ["1200-01-01", "1200-01-04"],
                ["1200-01-05", "1200-01-06"],
            ],
            dtype="datetime64[s]",
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=times)
        edges = np.array(
            ["1200-01-01", "1200-01-04", "1200-01-05"], dtype="datetime64[s]"
        )
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        # Cell 1 is fully covered by the first record; cell 2 [Jan 4, Jan 5)
        # contains no source data (there is a gap until Jan 5).
        np.testing.assert_allclose(result["v"].values, [1.0, np.nan])
        np.testing.assert_allclose(coverage["v"].values, [1.0, 0.0])
        # Output labels stay at second resolution rather than overflowing ns.
        assert result["time"].values[0] == np.datetime64("1200-01-01", "s")
        assert result["time"].dtype == np.dtype("datetime64[s]")

    def test_far_future_years_do_not_overflow(self):
        """Absolute timestamps reduced to ns can overflow int64; the shift avoids it."""
        times = np.array(["250000-01-01", "250000-01-05"], dtype="datetime64[s]")
        bounds = np.array(
            [
                ["250000-01-01", "250000-01-04"],
                ["250000-01-05", "250000-01-06"],
            ],
            dtype="datetime64[s]",
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=times)
        edges = np.array(
            ["249999-01-01", "250000-01-04", "250000-01-06"], dtype="datetime64[s]"
        )
        result, _ = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        np.testing.assert_allclose(result["v"].values, [1.0, 2.0])
        assert result["time"].dtype == np.dtype("datetime64[s]")
        assert result["time"].values[0] == np.datetime64("249999-01-01", "s")

    def test_mixed_numpy_resolutions_compare_exactly(self):
        times = np.array(["1200-01-01", "1200-01-05"], dtype="datetime64[s]")
        bounds = np.array(
            [
                ["1200-01-01", "1200-01-04"],
                ["1200-01-05", "1200-01-06"],
            ],
            dtype="datetime64[s]",
        )
        ds = _bounds_dataset([1.0, 2.0], bounds, times=times)
        edges = [
            np.datetime64("1199-12-01", "s"),
            np.datetime64("1200-01-07", "ms"),
            np.datetime64("1201-01-01", "us"),
        ]
        result, _ = ds.resample_interval(edges, kinds={"v": "sum"}, min_coverage=0.0)
        np.testing.assert_allclose(result["v"].values, [3.0, np.nan])

    @requires_cftime
    @pytest.mark.parametrize("datetime_type", ["DatetimeNoLeap", "Datetime360Day"])
    def test_cftime_calendars_use_real_lengths(self, datetime_type):
        cls = getattr(cftime, datetime_type)
        times = [cls(2000, 1, 1), cls(2000, 1, 2), cls(2000, 1, 4)]
        bounds = np.array(
            [
                [cls(2000, 1, 1), cls(2000, 1, 2)],
                [cls(2000, 1, 2), cls(2000, 1, 3)],
                [cls(2000, 1, 4), cls(2000, 1, 5)],
            ],
            dtype=object,
        )
        ds = xr.Dataset(
            {"v": ("time", [1.0, 2.0, 3.0])},
            coords={
                "time": ("time", times, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
            },
        )
        edges = [cls(2000, 1, 1), cls(2000, 1, 3), cls(2000, 1, 5)]
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "mean"}, min_coverage=0.0
        )
        # Cell 1 is fully covered: mean of 1 over one day and 2 over the next.
        # Cell 2 [Jan 3, Jan 5) only contains the Jan 4 record => 0.5 coverage.
        np.testing.assert_allclose(result["v"].values, [1.5, 3.0])
        np.testing.assert_allclose(coverage["v"].values, [1.0, 0.5])

    @requires_cftime
    def test_cross_calendar_source_bounds_rejected(self):
        cls = cftime.DatetimeNoLeap
        times = [cls(2000, 1, 1), cls(2000, 1, 3)]
        bounds = np.array(
            [
                [cls(2000, 1, 1), cls(2000, 1, 2)],
                [cls(2000, 1, 3), cls(2000, 1, 4)],
            ],
            dtype=object,
        )
        ds = xr.Dataset(
            {"v": ("time", [1.0, 2.0])},
            coords={
                "time": ("time", times, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
            },
        )
        with pytest.raises(ValueError, match="calendar"):
            ds.resample_interval(
                np.array(["2000-01-01", "2000-01-05"], dtype=D), kinds={"v": "sum"}
            )

    @requires_cftime
    def test_mixed_calendars_rejected(self):
        noleap = cftime.DatetimeNoLeap
        day360 = cftime.Datetime360Day
        times = [noleap(2000, 1, 1), day360(2000, 1, 3)]
        bounds = np.array(
            [
                [noleap(2000, 1, 1), noleap(2000, 1, 2)],
                [day360(2000, 1, 3), day360(2000, 1, 4)],
            ],
            dtype=object,
        )
        ds = xr.Dataset(
            {"v": ("time", [1.0, 2.0])},
            coords={
                "time": ("time", times, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
            },
        )
        with pytest.raises(ValueError, match="calendars"):
            ds.resample_interval(
                np.array([noleap(2000, 1, 1), noleap(2000, 1, 5)], dtype=object),
                kinds={"v": "sum"},
            )


class TestInvariants:
    def _irregular_dataset(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 40
        durations = np.tile(np.array([1, 2, 3, 4]), 10)
        starts = np.concatenate([[0], np.cumsum(durations)[:-1]])
        t0 = np.datetime64("2000-01-01T00:00:00", "s")
        bounds = np.stack(
            [
                t0 + starts.astype("timedelta64[s]"),
                t0 + (starts + durations).astype("timedelta64[s]"),
            ],
            axis=1,
        )
        times = bounds[:, 0]
        values = rng.normal(size=(3, n, 2))
        values[0, 5, 0] = np.nan
        values[1, 7, 1] = np.inf
        values[2, 20, 0] = -np.inf
        return xr.Dataset(
            {
                "v": (("x", "time", "y"), values),
                "w": ("time", rng.normal(size=n)),
            },
            coords={
                "time": ("time", times, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
                "x": [0, 1, 2],
                "y": ["a", "b"],
            },
        )

    def test_sum_conservation(self):
        ds = self._irregular_dataset()
        edges = np.array([ds["tb"].values[0, 0], ds["tb"].values[-1, 1]], dtype=object)
        result, _ = ds.resample_interval(
            np.array(list(edges)), kinds={"v": "sum", "w": "sum"}, min_coverage=0.0
        )
        expected_w = ds["w"].values.sum()
        np.testing.assert_allclose(result["w"].values, [expected_w])
        finite_v = np.where(np.isfinite(ds["v"].values), ds["v"].values, 0.0)
        expected_v = finite_v.sum(axis=1)
        np.testing.assert_allclose(result["v"].values[:, 0, :], expected_v)

    def test_mean_time_integral_conservation(self):
        ds = self._irregular_dataset()
        bounds = ds["tb"].values
        durations = (bounds[:, 1] - bounds[:, 0]).astype("timedelta64[s]").astype(float)
        values = ds["v"].values
        finite = np.isfinite(values)
        num = np.nansum(
            np.where(finite, values, 0.0) * durations[None, :, None], axis=1
        )
        den = np.sum(finite * durations[None, :, None], axis=1)
        expected = num / den

        edges = np.array([bounds[0, 0], bounds[-1, 1]], dtype=object)
        result, _ = ds.resample_interval(
            list(edges), kinds={"v": "mean", "w": "mean"}, min_coverage=0.0
        )
        np.testing.assert_allclose(result["v"].values[:, 0, :], expected)
        # The time integral of the mean over the *covered* duration equals the
        # time integral of the finite data.
        np.testing.assert_allclose(result["v"].values[:, 0, :] * den, num)

    def test_moving_labels_does_not_change_result(self):
        ds = self._irregular_dataset()
        edges = np.datetime64("2000-01-01T00:00:00", "s") + np.arange(0, 70, 7).astype(
            "timedelta64[s]"
        )
        first_result, first_coverage = ds.resample_interval(
            edges, kinds={"v": "mean", "w": "sum"}, min_coverage=0.3
        )
        bounds = ds["tb"].values
        shifted = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 3
        moved = ds.assign_coords(time=("time", shifted, ds["time"].attrs))
        second_result, second_coverage = moved.resample_interval(
            edges, kinds={"v": "mean", "w": "sum"}, min_coverage=0.3
        )
        for name in ("v", "w"):
            np.testing.assert_array_equal(
                first_result[name].values, second_result[name].values
            )
            np.testing.assert_array_equal(
                first_coverage[name].values, second_coverage[name].values
            )


class TestDask:
    @requires_dask
    def test_lazy_and_matches_numpy(self):
        import dask
        import dask.array as dask_array

        calls = []

        @dask.delayed(pure=False)
        def producer(index):
            calls.append(index)
            return np.arange(10, dtype=float) + index

        array = dask_array.concatenate(
            [dask_array.from_delayed(producer(i), (10,), dtype=float) for i in range(4)]
        )
        n = 40
        t0 = np.datetime64("2000-01-01", "D")
        times = t0 + np.arange(n).astype("timedelta64[D]")
        bounds = np.stack([times, times + np.timedelta64(1, "D")], axis=1)
        ds = xr.Dataset(
            {"v": ("time", array)},
            coords={
                "time": ("time", times, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
            },
        )
        edges = t0 + np.array([0, 14, 35], dtype="timedelta64[D]")
        result, coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        # No source tasks may have run yet.
        assert len(calls) == 0
        assert isinstance(result["v"].variable._data, dask_array.Array)
        assert isinstance(coverage["v"].variable._data, dask_array.Array)

        reference, reference_coverage = ds.resample_interval(
            edges, kinds={"v": "sum"}, min_coverage=0.0
        )
        np.testing.assert_allclose(result["v"].values, reference["v"].values)
        np.testing.assert_allclose(coverage["v"].values, reference_coverage["v"].values)
        assert len(calls) > 0

    @requires_dask
    def test_chunk_invariance_and_spatial_chunks(self):
        import dask.array as dask_array

        rng = np.random.default_rng(1)
        n = 24
        t0 = np.datetime64("2000-01-01", "s")
        starts = np.concatenate([[0], np.cumsum(np.tile([1, 2, 3], 8))[:-1]])
        bounds = np.stack(
            [
                t0 + starts.astype("timedelta64[s]"),
                t0 + (starts + np.tile([1, 2, 3], 8)).astype("timedelta64[s]"),
            ],
            axis=1,
        )
        times = bounds[:, 0]
        values = rng.normal(size=(3, n, 4))
        dataset = xr.Dataset(
            {"v": (("x", "time", "y"), values)},
            coords={
                "time": ("time", times, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
                "x": [0, 1, 2],
                "y": ["a", "b", "c", "d"],
            },
        )
        edges = t0 + np.arange(0, 50, 5).astype("timedelta64[s]")
        reference, _ = dataset.resample_interval(
            edges, kinds={"v": "mean"}, min_coverage=0.0
        )
        for time_chunks in (1, 4, 7, n):
            chunked = dataset.chunk({"time": time_chunks, "x": (1, 2), "y": (1, 3)})
            result, _ = chunked.resample_interval(
                edges, kinds={"v": "mean"}, min_coverage=0.0
            )
            assert isinstance(result["v"].variable._data, dask_array.Array)
            # Existing spatial chunking is preserved.
            assert result["v"].variable._data.chunks[0] == (1, 2)
            assert result["v"].variable._data.chunks[2] == (1, 3)
            actual = result["v"].values
            np.testing.assert_allclose(actual, reference["v"].values)
            assert np.array_equal(np.isnan(actual), np.isnan(reference["v"].values))

    @requires_dask
    def test_large_adjacent_grids_do_not_materialize_dense_matrix(self):
        n_sources = 20_000
        n_targets = 20_000
        t0 = np.datetime64("2000-01-01", "s")
        starts = np.arange(n_sources, dtype="timedelta64[s]")
        bounds = np.stack([t0 + starts, t0 + starts + np.timedelta64(1, "s")], axis=1)
        dataset = xr.Dataset(
            {"v": ("time", np.arange(n_sources, dtype=float))},
            coords={
                "time": ("time", t0 + starts, {"bounds": "tb"}),
                "tb": (("time", "nv"), bounds),
            },
        ).chunk({"time": 100})
        edges = t0 + np.arange(n_targets, dtype="timedelta64[s]")
        result, _ = dataset.resample_interval(edges, kinds={"v": "sum"})
        # The dask graph grows with actual overlap (a few tens of thousands of
        # nodes), not with n_sources * n_targets.
        assert len(result["v"].variable._data.dask) < 100 * n_targets
        np.testing.assert_allclose(result["v"].values[:3], [0.0, 1.0, 2.0])
