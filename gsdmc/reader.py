"""NetCDF wind-field reader.

Reads 3-D (or 4-D with a time dimension) wind-field data from a NetCDF file
and returns the wind-speed magnitude on a regular grid together with the
physical coordinate arrays.

Supported layouts
-----------------
* ERA5 style  – variables ``u``, ``v``, ``w`` (or ``u10``, ``v10`` for
  10-m wind); dimensions ``(time, level, latitude, longitude)`` or
  ``(level, latitude, longitude)``.
* WRF style   – variables ``U``, ``V``, ``W``; dimensions
  ``(Time, bottom_top, south_north, west_east)``.
* Generic     – caller can specify variable names and dimension order
  explicitly via keyword arguments.

The reader always collapses the time axis by taking a single time step
(defaulting to index 0) and returns a ``(nz, ny, nx)`` float64 array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import netCDF4 as nc4  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "netCDF4 is required by gsdmc.reader. "
        "Install it with:  pip install netCDF4"
    ) from exc


# ---------------------------------------------------------------------------
# Default variable/dimension name candidates (tried in order)
# ---------------------------------------------------------------------------

_U_CANDIDATES = ("u", "U", "ua", "u_component_of_wind", "UGRD")
_V_CANDIDATES = ("v", "V", "va", "v_component_of_wind", "VGRD")
_W_CANDIDATES = ("w", "W", "wa", "vertical_velocity", "omega", "VVEL")

_LAT_CANDIDATES = ("lat", "latitude", "XLAT", "south_north")
_LON_CANDIDATES = ("lon", "longitude", "XLONG", "west_east")
_LEV_CANDIDATES = (
    "level",
    "lev",
    "pressure",
    "plev",
    "bottom_top",
    "height",
    "altitude",
    "z",
)
_TIME_CANDIDATES = ("time", "Time", "t")


def _first_match(ds: "nc4.Dataset", candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _squeeze_time(arr: np.ndarray, time_idx: int) -> np.ndarray:
    """Drop the time dimension if present (assumed to be axis 0)."""
    if arr.ndim == 4:
        return arr[time_idx]
    return arr  # already 3-D


class WindFieldData:
    """Container returned by :func:`read_wind_field`.

    Attributes
    ----------
    speed : ndarray, shape (nz, ny, nx)
        Wind-speed magnitude ``sqrt(u² + v² + w²)`` in m s⁻¹.
    u, v, w : ndarray or None
        Raw wind components on the same grid.  *w* is ``None`` when no
        vertical component is found in the file.
    lat : ndarray, shape (ny,) or (ny, nx)
    lon : ndarray, shape (nx,) or (ny, nx)
    lev : ndarray, shape (nz,) or None
    spacing : tuple (dz, dy, dx)
        Approximate uniform grid spacing derived from the coordinate arrays.
        Falls back to ``(1, 1, 1)`` when coordinates are unavailable.
    origin : tuple (z0, y0, x0)
        Physical coordinates of the ``(0, 0, 0)`` grid point.
    """

    def __init__(
        self,
        speed: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray | None,
        lat: np.ndarray | None,
        lon: np.ndarray | None,
        lev: np.ndarray | None,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
    ):
        self.speed = speed
        self.u = u
        self.v = v
        self.w = w
        self.lat = lat
        self.lon = lon
        self.lev = lev
        self.spacing = spacing
        self.origin = origin

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WindFieldData(shape={self.speed.shape}, "
            f"speed=[{self.speed.min():.2f}, {self.speed.max():.2f}] m/s)"
        )


def _infer_spacing(coord: np.ndarray | None) -> tuple[float, float]:
    """Return (mean_step, first_value) from a 1-D coordinate array."""
    if coord is None or coord.ndim != 1 or len(coord) < 2:
        return 1.0, 0.0
    step = float(np.mean(np.diff(coord.astype(np.float64))))
    return abs(step), float(coord.flat[0])


def read_wind_field(
    path: str | Path,
    u_var: str | None = None,
    v_var: str | None = None,
    w_var: str | None = None,
    time_idx: int = 0,
    fill_value: float = 0.0,
) -> "WindFieldData":
    """Read a 3-D wind field from a NetCDF file.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF (.nc / .nc4) file.
    u_var, v_var, w_var : str, optional
        Variable names to use for the eastward, northward, and vertical wind
        components.  When omitted, common names are tried automatically.
    time_idx : int, optional
        Which time step to load when the file contains a time dimension
        (default 0).
    fill_value : float, optional
        Value to substitute for masked / fill-value cells (default 0).

    Returns
    -------
    WindFieldData

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    KeyError
        When the required wind-component variables cannot be found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    with nc4.Dataset(str(path), "r") as ds:
        # -- Locate wind component variables --------------------------------
        u_name = u_var or _first_match(ds, _U_CANDIDATES)
        v_name = v_var or _first_match(ds, _V_CANDIDATES)
        if u_name is None or v_name is None:
            raise KeyError(
                "Could not find U/V wind variables in the file. "
                f"Available variables: {list(ds.variables.keys())}"
            )
        w_name = w_var or _first_match(ds, _W_CANDIDATES)

        def _load(name: str) -> np.ndarray:
            raw = ds.variables[name][:]
            if hasattr(raw, "filled"):
                raw = raw.filled(fill_value)
            arr = np.asarray(raw, dtype=np.float64)
            return _squeeze_time(arr, time_idx)

        u_arr = _load(u_name)
        v_arr = _load(v_name)
        w_arr = _load(w_name) if w_name is not None else None

        # -- Coordinate variables -------------------------------------------
        lat_name = _first_match(ds, _LAT_CANDIDATES)
        lon_name = _first_match(ds, _LON_CANDIDATES)
        lev_name = _first_match(ds, _LEV_CANDIDATES)

        def _load_coord(name: str | None) -> np.ndarray | None:
            if name is None:
                return None
            raw = ds.variables[name][:]
            if hasattr(raw, "filled"):
                raw = raw.filled(np.nan)
            return np.asarray(raw, dtype=np.float64)

        lat = _load_coord(lat_name)
        lon = _load_coord(lon_name)
        lev = _load_coord(lev_name)

        # Squeeze 1-D lat/lon if they were loaded with extra dims
        if lat is not None and lat.ndim > 1:
            lat = lat[:, 0] if lat.shape[1] > 1 else lat.ravel()
        if lon is not None and lon.ndim > 1:
            lon = lon[0, :] if lon.shape[0] > 1 else lon.ravel()

    # -- Compute wind speed -------------------------------------------------
    from .wind import compute_wind_speed  # avoid circular at module level

    speed = compute_wind_speed(u_arr, v_arr, w_arr)

    # -- Infer grid spacing / origin ----------------------------------------
    dz, z0 = _infer_spacing(lev)
    # lat may increase or decrease – use abs(step)
    lat_step, lat0 = _infer_spacing(lat)
    lon_step, lon0 = _infer_spacing(lon)

    spacing = (dz, lat_step, lon_step)
    origin = (z0, lat0, lon0)

    return WindFieldData(
        speed=speed,
        u=u_arr,
        v=v_arr,
        w=w_arr,
        lat=lat,
        lon=lon,
        lev=lev,
        spacing=spacing,
        origin=origin,
    )
