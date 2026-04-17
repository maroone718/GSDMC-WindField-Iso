"""Wind-speed computation from vector wind components.

Supports both 2-D (horizontal only) and 3-D (including vertical) fields.
"""

from __future__ import annotations

import numpy as np


def compute_wind_speed(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray | None = None,
) -> np.ndarray:
    """Compute wind-speed magnitude from component arrays.

    Parameters
    ----------
    u : ndarray
        Eastward (or x-direction) wind component.  Any shape is accepted as
        long as *u*, *v* (and optionally *w*) are broadcast-compatible.
    v : ndarray
        Northward (or y-direction) wind component.
    w : ndarray, optional
        Vertical (or z-direction) wind component.  When omitted the
        horizontal wind speed ``sqrt(u² + v²)`` is returned.

    Returns
    -------
    speed : ndarray
        Wind-speed magnitude with the same shape as the broadcasted inputs.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([3.0, 0.0])
    >>> v = np.array([4.0, 0.0])
    >>> compute_wind_speed(u, v)
    array([5., 0.])
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    speed_sq = u * u + v * v
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        speed_sq = speed_sq + w * w
    return np.sqrt(speed_sq)


def wind_speed_percentile(
    speed: np.ndarray,
    percentile: float,
    mask: np.ndarray | None = None,
) -> float:
    """Compute the *p*-th percentile wind speed (useful for choosing isovalues).

    Parameters
    ----------
    speed : ndarray
        Wind-speed array (any shape).
    percentile : float
        Value in [0, 100].
    mask : ndarray of bool, optional
        Boolean mask selecting the grid points to include.  By default all
        finite (non-NaN) grid points are used.

    Returns
    -------
    float
        Percentile value of the wind speed.
    """
    s = np.asarray(speed, dtype=np.float64).ravel()
    if mask is not None:
        s = s[np.asarray(mask).ravel().astype(bool)]
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite values found in speed array")
    return float(np.percentile(s, percentile))
