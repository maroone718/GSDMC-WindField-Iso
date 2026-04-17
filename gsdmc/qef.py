"""Quadratic Error Function (QEF) solver for dual-vertex positioning.

The QEF is the sum of squared distances from a point *x* to a set of planes,
each plane defined by a surface point *p_i* and a unit normal *n_i*::

    E(x) = sum_i  ( n_i · (x - p_i) )^2

Minimising this is a linear least-squares problem:
    A x = b,   A[i] = n_i,   b[i] = n_i · p_i
"""

import numpy as np


def solve_qef(
    intersection_points,
    intersection_normals,
    mass_point=None,
    regularisation=1e-4,
):
    """Solve the QEF and return the optimal vertex position.

    Parameters
    ----------
    intersection_points : array_like, shape (K, 3)
        Points on the isosurface where primal edges cross it.
    intersection_normals : array_like, shape (K, 3)
        Unit surface normals at the corresponding intersection points.
    mass_point : ndarray, shape (3,), optional
        Fallback / regularisation point used when the system is rank-deficient
        (e.g. a planar or edge feature). Defaults to the mean of the
        intersection points.
    regularisation : float, optional
        Weight of the Tikhonov regularisation that biases the solution
        toward *mass_point*.  A small value (default 1e-4) keeps the vertex
        close to the feature while avoiding numerical instability.

    Returns
    -------
    x : ndarray, shape (3,)
        Optimal vertex position.
    """
    pts = np.asarray(intersection_points, dtype=np.float64)
    nrms = np.asarray(intersection_normals, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("intersection_points must have shape (K, 3)")
    if nrms.shape != pts.shape:
        raise ValueError("intersection_normals must have the same shape as intersection_points")

    K = len(pts)
    if K == 0:
        return np.asarray(mass_point, dtype=np.float64) if mass_point is not None else np.zeros(3)

    mp = np.mean(pts, axis=0) if mass_point is None else np.asarray(mass_point, dtype=np.float64)

    # QEF system: A x = b
    A = nrms  # (K, 3)
    b = np.einsum("ki,ki->k", nrms, pts)  # dot(n_i, p_i)

    # Tikhonov regularisation: augment with (sqrt(reg) * I) x = sqrt(reg) * mp
    sqrt_reg = np.sqrt(regularisation)
    A_aug = np.vstack([A, sqrt_reg * np.eye(3)])
    b_aug = np.concatenate([b, sqrt_reg * mp])

    x, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return x
