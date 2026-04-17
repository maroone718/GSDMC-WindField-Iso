"""Generalized Symmetric Dual Marching Cubes (GSDMC) isosurface extraction.

Algorithm overview
------------------
The Dual Marching Cubes algorithm (Schaefer & Warren, 2004) generates
isosurfaces by placing one vertex *inside* each active grid cell rather than
on cell edges.  The "Generalized Symmetric" variant uses a Quadratic Error
Function (QEF) minimisation to find the optimal vertex position, which makes
the mesh symmetric with respect to inside/outside swaps and handles all 256
topological cases without special casing.

Pipeline
~~~~~~~~
1. Classify every grid vertex as *inside* (field ≥ isovalue) or *outside*.
2. For every primal cell whose 8 corners are *not* all the same sign
   (an "active" cell), solve the QEF to place a single dual vertex inside
   the cell.
3. For every active primal edge (the two endpoint vertices have opposite
   signs), the four cells that share that edge each contribute one dual
   vertex; connect these four vertices into two triangles (a quad split
   along the shorter diagonal).

Public API
----------
:func:`extract_isosurface` – main entry point.
"""

from __future__ import annotations

import numpy as np

from .qef import solve_qef
from .tables import (
    CUBE_EDGES,
    CUBE_VERTS,
    X_EDGE_CELL_OFFSETS,
    Y_EDGE_CELL_OFFSETS,
    Z_EDGE_CELL_OFFSETS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_gradient(field: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Estimate the gradient of *field* using central finite differences.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)
    spacing : (dx, dy, dz)

    Returns
    -------
    grad : ndarray, shape (3, nx, ny, nz)
        grad[0] = ∂f/∂x,  grad[1] = ∂f/∂y,  grad[2] = ∂f/∂z
    """
    dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    f = field.astype(np.float64)
    grad = np.zeros((3,) + f.shape, dtype=np.float64)
    nx, ny, nz = f.shape

    # i-direction (axis 0)
    if nx >= 3:
        grad[0, 1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
    if nx >= 2:
        grad[0, 0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        grad[0, -1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx

    # j-direction (axis 1)
    if ny >= 3:
        grad[1, :, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
    if ny >= 2:
        grad[1, :, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        grad[1, :, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy

    # k-direction (axis 2)
    if nz >= 3:
        grad[2, :, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
    if nz >= 2:
        grad[2, :, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
        grad[2, :, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz

    return grad


def _add_quads(
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_k: np.ndarray,
    v0_inside: np.ndarray,
    cell_vertex_idx: np.ndarray,
    offsets: np.ndarray,
    fixed_axis: int,
    ncy: int,
    ncz: int,
    ncx: int,
) -> list[list[int]]:
    """Build triangles for all active edges along one axis direction.

    Parameters
    ----------
    edge_i, edge_j, edge_k : 1-D int arrays
        Grid indices of the *lesser-coordinate* endpoint of each active edge.
    v0_inside : 1-D bool array
        True when the lesser-coordinate endpoint is inside the isosurface.
    cell_vertex_idx : ndarray, shape (ncx, ncy, ncz)
        Maps cell grid index to dual-vertex output index; -1 means inactive.
    offsets : ndarray, shape (4, 2)
        For the two free coordinates of the adjacent cells (see ``tables``).
    fixed_axis : int
        0 for X-edges, 1 for Y-edges, 2 for Z-edges.
    ncy, ncz, ncx : int
        Number of cells in each direction.

    Returns
    -------
    triangles : list of [v0, v1, v2]
    """
    triangles: list[list[int]] = []

    for idx in range(len(edge_i)):
        ei, ej, ek = int(edge_i[idx]), int(edge_j[idx]), int(edge_k[idx])
        inside_v0 = bool(v0_inside[idx])

        quad: list[int] = []
        for d0, d1 in offsets:
            d0, d1 = int(d0), int(d1)
            if fixed_axis == 0:      # X-edge: free axes are j, k
                ci, cj, ck = ei, ej + d0, ek + d1
            elif fixed_axis == 1:    # Y-edge: free axes are i, k
                ci, cj, ck = ei + d0, ej, ek + d1
            else:                    # Z-edge: free axes are i, j
                ci, cj, ck = ei + d0, ej + d1, ek

            if 0 <= ci < ncx and 0 <= cj < ncy and 0 <= ck < ncz:
                v = int(cell_vertex_idx[ci, cj, ck])
                if v >= 0:
                    quad.append(v)

        if len(quad) != 4:
            # Boundary edge – not enough surrounding cells; skip
            continue

        a, b, c, d = quad
        if inside_v0:
            # v0 is inside → outward normal points in POSITIVE axis direction.
            # Use CCW order when looking in +axis (our stored order).
            triangles.append([a, b, c])
            triangles.append([a, c, d])
        else:
            # v0 is outside → outward normal points in NEGATIVE axis direction.
            # Reverse winding.
            triangles.append([a, c, b])
            triangles.append([a, d, c])

    return triangles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_isosurface(
    scalar_field: np.ndarray,
    isovalue: float,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_qef: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an isosurface using Generalized Symmetric Dual Marching Cubes.

    Parameters
    ----------
    scalar_field : ndarray, shape (nx, ny, nz)
        3-D scalar field sampled on a regular grid.  For wind-field data this
        is typically the wind-speed magnitude ``sqrt(u² + v² + w²)``.
    isovalue : float
        The threshold value that defines the isosurface.  Grid points with
        ``scalar_field >= isovalue`` are considered *inside*.
    spacing : (dx, dy, dz), optional
        Physical cell size in each direction.  Defaults to ``(1, 1, 1)``.
    origin : (ox, oy, oz), optional
        Physical coordinates of grid point ``(0, 0, 0)``.
    use_qef : bool, optional
        If *True* (default) use QEF minimisation for precise vertex placement.
        If *False* use the arithmetic mean of edge intersection points (faster
        but less accurate for sharp features).

    Returns
    -------
    vertices : ndarray, shape (N, 3)
        Vertex positions in physical coordinates.
    triangles : ndarray, shape (M, 3), dtype int32
        Triangle faces given as triplets of vertex indices.
    vertex_normals : ndarray, shape (N, 3)
        Outward unit normals at each vertex (averaged from edge normals).

    Notes
    -----
    The isosurface is empty (``vertices`` and ``triangles`` will have length
    0) when the *isovalue* lies outside the range of *scalar_field*.
    """
    field = np.asarray(scalar_field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("scalar_field must be a 3-D array")

    nx, ny, nz = field.shape
    dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

    # Physical coordinate arrays
    cx = ox + np.arange(nx) * dx
    cy = oy + np.arange(ny) * dy
    cz = oz + np.arange(nz) * dz

    # Boolean inside/outside classification
    inside = field >= isovalue  # (nx, ny, nz)

    # Gradient used for surface normals at intersection points
    grad = _compute_gradient(field, (dx, dy, dz))  # (3, nx, ny, nz)

    # ------------------------------------------------------------------
    # Step 1 – find active cells
    # ------------------------------------------------------------------
    ncx, ncy, ncz = nx - 1, ny - 1, nz - 1
    if ncx <= 0 or ncy <= 0 or ncz <= 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros((0, 3), dtype=np.int32), empty

    # Build the 8-bit case index for every cell vectorised
    case_bits = np.zeros((ncx, ncy, ncz), dtype=np.uint8)
    for bit, (di, dj, dk) in enumerate(CUBE_VERTS):
        case_bits |= inside[di : ncx + di, dj : ncy + dj, dk : ncz + dk].view(np.uint8) << bit

    active_mask = (case_bits != 0) & (case_bits != 255)
    active_cells = np.argwhere(active_mask)  # (N_active, 3)
    n_active = len(active_cells)

    if n_active == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros((0, 3), dtype=np.int32), empty

    # ------------------------------------------------------------------
    # Step 2 – compute one dual vertex per active cell
    # ------------------------------------------------------------------
    vertices = np.empty((n_active, 3), dtype=np.float64)
    vertex_normals = np.zeros((n_active, 3), dtype=np.float64)

    # Fast lookup: cell (ci, cj, ck) → vertex index; -1 means inactive
    cell_vertex_idx = np.full((ncx, ncy, ncz), -1, dtype=np.int32)
    cell_vertex_idx[active_cells[:, 0], active_cells[:, 1], active_cells[:, 2]] = np.arange(
        n_active, dtype=np.int32
    )

    for vidx, (ci, cj, ck) in enumerate(active_cells):
        ci, cj, ck = int(ci), int(cj), int(ck)

        qef_pts: list[np.ndarray] = []
        qef_nrms: list[np.ndarray] = []

        for ea, eb in CUBE_EDGES:
            dai, daj, dak = CUBE_VERTS[ea]
            dbi, dbj, dbk = CUBE_VERTS[eb]
            gia, gja, gka = ci + dai, cj + daj, ck + dak
            gib, gjb, gkb = ci + dbi, cj + dbj, ck + dbk

            a_in = bool(inside[gia, gja, gka])
            b_in = bool(inside[gib, gjb, gkb])
            if a_in == b_in:
                continue  # edge not active

            fa = field[gia, gja, gka]
            fb = field[gib, gjb, gkb]
            denom = fb - fa
            t = (isovalue - fa) / denom if abs(denom) > 1e-15 else 0.5

            pa = np.array([cx[gia], cy[gja], cz[gka]])
            pb = np.array([cx[gib], cy[gjb], cz[gkb]])
            p = pa + t * (pb - pa)

            na = grad[:, gia, gja, gka]
            nb = grad[:, gib, gjb, gkb]
            n = na + t * (nb - na)
            n_len = float(np.linalg.norm(n))
            if n_len > 1e-12:
                n = n / n_len
            else:
                edge_dir = pb - pa
                e_len = float(np.linalg.norm(edge_dir))
                n = edge_dir / e_len if e_len > 1e-12 else np.array([0.0, 0.0, 1.0])

            qef_pts.append(p)
            qef_nrms.append(n)

        if not qef_pts:
            # Degenerate active cell – fall back to cell centre
            vertices[vidx] = np.array(
                [cx[ci] + dx * 0.5, cy[cj] + dy * 0.5, cz[ck] + dz * 0.5]
            )
            vertex_normals[vidx] = np.array([0.0, 0.0, 1.0])
            continue

        mass_point = np.mean(qef_pts, axis=0)

        if use_qef:
            v = solve_qef(qef_pts, qef_nrms, mass_point)
            # Clamp to cell bounding box to avoid out-of-cell vertices
            v[0] = float(np.clip(v[0], cx[ci], cx[ci + 1]))
            v[1] = float(np.clip(v[1], cy[cj], cy[cj + 1]))
            v[2] = float(np.clip(v[2], cz[ck], cz[ck + 1]))
        else:
            v = mass_point

        vertices[vidx] = v

        avg_n = np.mean(qef_nrms, axis=0)
        n_len = float(np.linalg.norm(avg_n))
        vertex_normals[vidx] = avg_n / n_len if n_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # Step 3 – build the dual mesh from active primal edges
    # ------------------------------------------------------------------
    triangles: list[list[int]] = []

    # ---- X-edges ----
    x_active = inside[:-1, :, :] != inside[1:, :, :]
    xi, xj, xk = np.where(x_active)
    v0_inside_x = inside[:-1, :, :][x_active]
    triangles += _add_quads(xi, xj, xk, v0_inside_x, cell_vertex_idx,
                             X_EDGE_CELL_OFFSETS, 0, ncy, ncz, ncx)

    # ---- Y-edges ----
    y_active = inside[:, :-1, :] != inside[:, 1:, :]
    yi, yj, yk = np.where(y_active)
    v0_inside_y = inside[:, :-1, :][y_active]
    triangles += _add_quads(yi, yj, yk, v0_inside_y, cell_vertex_idx,
                             Y_EDGE_CELL_OFFSETS, 1, ncy, ncz, ncx)

    # ---- Z-edges ----
    z_active = inside[:, :, :-1] != inside[:, :, 1:]
    zi, zj, zk = np.where(z_active)
    v0_inside_z = inside[:, :, :-1][z_active]
    triangles += _add_quads(zi, zj, zk, v0_inside_z, cell_vertex_idx,
                             Z_EDGE_CELL_OFFSETS, 2, ncy, ncz, ncx)

    tris_arr = (
        np.array(triangles, dtype=np.int32)
        if triangles
        else np.zeros((0, 3), dtype=np.int32)
    )

    return vertices, tris_arr, vertex_normals
