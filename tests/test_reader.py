"""Tests for gsdmc.reader – NetCDF wind-field reader.

Uses the netCDF4 library to create temporary in-memory / on-disk test files
rather than requiring a real ERA5 or WRF dataset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gsdmc.reader import WindFieldData, read_wind_field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nc(
    path: str,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray | None = None,
    u_name: str = "u",
    v_name: str = "v",
    w_name: str = "w",
    add_coords: bool = True,
):
    """Create a minimal NetCDF file with wind components."""
    nz, ny, nx = u.shape
    import netCDF4 as nc4

    ds = nc4.Dataset(path, "w")
    ds.createDimension("z", nz)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    uu = ds.createVariable(u_name, "f4", ("z", "y", "x"))
    vv = ds.createVariable(v_name, "f4", ("z", "y", "x"))
    uu[:] = u
    vv[:] = v

    if w is not None:
        ww = ds.createVariable(w_name, "f4", ("z", "y", "x"))
        ww[:] = w

    if add_coords:
        lat_var = ds.createVariable("lat", "f4", ("y",))
        lon_var = ds.createVariable("lon", "f4", ("x",))
        lev_var = ds.createVariable("level", "f4", ("z",))
        lat_var[:] = np.linspace(0.0, 10.0, ny)
        lon_var[:] = np.linspace(100.0, 110.0, nx)
        lev_var[:] = np.arange(1, nz + 1) * 100.0

    ds.close()


def _make_nc_with_time(path: str, u: np.ndarray, v: np.ndarray):
    """Create a NetCDF file with a time dimension prepended."""
    import netCDF4 as nc4
    nt, nz, ny, nx = 3, *u.shape
    ds = nc4.Dataset(path, "w")
    ds.createDimension("time", nt)
    ds.createDimension("z", nz)
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    uu = ds.createVariable("u", "f4", ("time", "z", "y", "x"))
    vv = ds.createVariable("v", "f4", ("time", "z", "y", "x"))
    for t in range(nt):
        uu[t] = u * (t + 1)
        vv[t] = v * (t + 1)

    time_var = ds.createVariable("time", "f4", ("time",))
    time_var[:] = np.arange(nt, dtype=np.float32)
    ds.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_nc(tmp_path):
    """A 3×4×5 wind field (nz=3, ny=4, nx=5) without W component."""
    rng = np.random.default_rng(0)
    u = rng.uniform(-10, 10, (3, 4, 5)).astype(np.float32)
    v = rng.uniform(-10, 10, (3, 4, 5)).astype(np.float32)
    path = str(tmp_path / "wind_simple.nc")
    _make_nc(path, u, v, w=None)
    return path, u, v


@pytest.fixture()
def full_3d_nc(tmp_path):
    """A 3×4×5 wind field with U, V, W components."""
    rng = np.random.default_rng(1)
    u = rng.uniform(-10, 10, (3, 4, 5)).astype(np.float32)
    v = rng.uniform(-10, 10, (3, 4, 5)).astype(np.float32)
    w = rng.uniform(-2, 2, (3, 4, 5)).astype(np.float32)
    path = str(tmp_path / "wind_3d.nc")
    _make_nc(path, u, v, w=w)
    return path, u, v, w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadWindField:
    def test_returns_wind_field_data(self, simple_nc):
        path, u, v = simple_nc
        data = read_wind_field(path)
        assert isinstance(data, WindFieldData)

    def test_speed_shape(self, simple_nc):
        path, u, v = simple_nc
        data = read_wind_field(path)
        assert data.speed.shape == (3, 4, 5)

    def test_speed_values_correct(self, simple_nc):
        path, u, v = simple_nc
        data = read_wind_field(path)
        expected = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
        np.testing.assert_allclose(data.speed, expected, rtol=1e-5)

    def test_w_none_when_missing(self, simple_nc):
        path, u, v = simple_nc
        data = read_wind_field(path)
        assert data.w is None

    def test_w_present_when_provided(self, full_3d_nc):
        path, u, v, w = full_3d_nc
        data = read_wind_field(path)
        assert data.w is not None
        assert data.w.shape == w.shape

    def test_speed_includes_w(self, full_3d_nc):
        path, u, v, w = full_3d_nc
        data = read_wind_field(path)
        expected = np.sqrt(
            u.astype(np.float64) ** 2
            + v.astype(np.float64) ** 2
            + w.astype(np.float64) ** 2
        )
        np.testing.assert_allclose(data.speed, expected, rtol=1e-5)

    def test_coordinate_arrays_present(self, simple_nc):
        path, _, _ = simple_nc
        data = read_wind_field(path)
        assert data.lat is not None
        assert data.lon is not None
        assert data.lev is not None

    def test_spacing_positive(self, simple_nc):
        path, _, _ = simple_nc
        data = read_wind_field(path)
        assert all(s > 0 for s in data.spacing)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_wind_field(str(tmp_path / "nonexistent.nc"))

    def test_missing_uv_raises(self, tmp_path):
        import netCDF4 as nc4
        path = str(tmp_path / "no_wind.nc")
        ds = nc4.Dataset(path, "w")
        ds.createDimension("x", 3)
        ds.createVariable("temperature", "f4", ("x",))[:] = [280, 290, 300]
        ds.close()
        with pytest.raises(KeyError):
            read_wind_field(path)

    def test_explicit_variable_names(self, tmp_path):
        rng = np.random.default_rng(5)
        u = rng.standard_normal((2, 3, 4)).astype(np.float32)
        v = rng.standard_normal((2, 3, 4)).astype(np.float32)
        path = str(tmp_path / "custom_vars.nc")
        _make_nc(path, u, v, u_name="ucomp", v_name="vcomp", add_coords=False)
        data = read_wind_field(path, u_var="ucomp", v_var="vcomp")
        assert data.speed.shape == (2, 3, 4)

    def test_time_dimension_squeezed(self, tmp_path):
        rng = np.random.default_rng(9)
        u = rng.standard_normal((2, 3, 4)).astype(np.float32)
        v = rng.standard_normal((2, 3, 4)).astype(np.float32)
        path = str(tmp_path / "time_nc.nc")
        _make_nc_with_time(path, u, v)
        data = read_wind_field(path, time_idx=0)
        assert data.speed.shape == (2, 3, 4)

    def test_time_idx_selection(self, tmp_path):
        rng = np.random.default_rng(3)
        u = rng.standard_normal((2, 3, 4)).astype(np.float32)
        v = rng.standard_normal((2, 3, 4)).astype(np.float32)
        path = str(tmp_path / "time_nc2.nc")
        _make_nc_with_time(path, u, v)
        data0 = read_wind_field(path, time_idx=0)
        data1 = read_wind_field(path, time_idx=1)
        # time_idx=1 → u is 2×, v is 2×, so speed is 2× that of time_idx=0
        np.testing.assert_allclose(data1.speed, data0.speed * 2.0, rtol=1e-4)
