"""Tests for gsdmc.core – GSDMC isosurface extraction."""

import numpy as np
import pytest

from gsdmc.core import extract_isosurface


def _sphere_field(radius: float = 3.0, grid: int = 20):
    """Return a scalar field equal to sqrt(x² + y² + z²) on a cube grid."""
    half = grid // 2
    x, y, z = np.mgrid[-half:half, -half:half, -half:half].astype(np.float64)
    return np.sqrt(x**2 + y**2 + z**2)


def _plane_field(grid: int = 10, axis: int = 0):
    """Return a scalar field that increases linearly along *axis*."""
    shape = [grid, grid, grid]
    coords = np.arange(grid, dtype=np.float64)
    field = np.zeros(shape)
    slices = [None, None, None]
    slices[axis] = slice(None)
    field[:] = coords[tuple(slices)]
    return field


class TestExtractIsosurface:
    # ------------------------------------------------------------------
    # Basic sanity
    # ------------------------------------------------------------------

    def test_returns_three_arrays(self):
        field = _sphere_field()
        result = extract_isosurface(field, isovalue=5.0)
        assert len(result) == 3

    def test_output_shapes_consistent(self):
        field = _sphere_field()
        verts, tris, normals = extract_isosurface(field, isovalue=5.0)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert tris.ndim == 2 and tris.shape[1] == 3
        assert normals.shape == verts.shape

    def test_triangle_indices_in_range(self):
        field = _sphere_field()
        verts, tris, _ = extract_isosurface(field, isovalue=5.0)
        if len(tris) > 0:
            assert tris.min() >= 0
            assert tris.max() < len(verts)

    def test_nonempty_surface(self):
        """A sphere field at an interior isovalue should produce geometry."""
        field = _sphere_field(grid=20)
        verts, tris, _ = extract_isosurface(field, isovalue=5.0)
        assert len(verts) > 0
        assert len(tris) > 0

    # ------------------------------------------------------------------
    # Edge cases – trivially inside / outside
    # ------------------------------------------------------------------

    def test_all_inside_produces_empty(self):
        """All vertices above isovalue → no surface."""
        field = np.ones((8, 8, 8)) * 10.0
        verts, tris, normals = extract_isosurface(field, isovalue=5.0)
        assert len(verts) == 0
        assert len(tris) == 0

    def test_all_outside_produces_empty(self):
        """All vertices below isovalue → no surface."""
        field = np.ones((8, 8, 8)) * 1.0
        verts, tris, normals = extract_isosurface(field, isovalue=5.0)
        assert len(verts) == 0

    def test_isovalue_above_range_produces_empty(self):
        field = _sphere_field()
        verts, tris, _ = extract_isosurface(field, isovalue=9999.0)
        assert len(verts) == 0

    def test_isovalue_below_range_produces_empty(self):
        field = _sphere_field()
        verts, tris, _ = extract_isosurface(field, isovalue=-1.0)
        assert len(verts) == 0

    def test_tiny_grid_no_crash(self):
        """2×2×2 grid should not raise."""
        field = np.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        extract_isosurface(field, isovalue=3.5)  # should not raise

    def test_grid_too_small_no_crash(self):
        """1×1×1 grid has no cells – returns empty."""
        field = np.ones((1, 1, 1))
        verts, tris, normals = extract_isosurface(field, isovalue=0.5)
        assert len(verts) == 0

    # ------------------------------------------------------------------
    # Geometry correctness
    # ------------------------------------------------------------------

    def test_plane_isosurface(self):
        """A linear scalar field crossing the isovalue creates a flat surface."""
        field = _plane_field(grid=10, axis=0)  # f(i,j,k) = i in [0,9]
        isovalue = 4.5
        verts, tris, _ = extract_isosurface(field, isovalue=isovalue)
        # All vertices should be near x = 4.5
        if len(verts) > 0:
            np.testing.assert_allclose(verts[:, 0], 4.5, atol=0.51)

    def test_spacing_scales_output(self):
        """Grid spacing should be reflected in vertex coordinates."""
        field = _sphere_field(grid=12)
        verts_unit, _, _ = extract_isosurface(field, isovalue=3.0, spacing=(1, 1, 1))
        verts_scaled, _, _ = extract_isosurface(field, isovalue=3.0, spacing=(2, 2, 2))
        if len(verts_unit) > 0 and len(verts_scaled) > 0:
            assert verts_scaled[:, 0].max() > verts_unit[:, 0].max()

    def test_origin_shifts_output(self):
        """Origin parameter shifts all vertex coordinates."""
        field = _sphere_field(grid=12)
        shift = (10.0, 20.0, 30.0)
        verts_orig, _, _ = extract_isosurface(field, isovalue=3.0, origin=(0, 0, 0))
        verts_shift, _, _ = extract_isosurface(field, isovalue=3.0, origin=shift)
        if len(verts_orig) > 0 and len(verts_shift) > 0:
            diff = verts_shift - verts_orig
            np.testing.assert_allclose(diff.mean(axis=0), shift, atol=0.5)

    # ------------------------------------------------------------------
    # QEF vs. average-point mode
    # ------------------------------------------------------------------

    def test_average_mode_returns_valid_mesh(self):
        field = _sphere_field()
        verts, tris, normals = extract_isosurface(field, isovalue=5.0, use_qef=False)
        assert verts.shape[1] == 3
        if len(tris) > 0:
            assert tris.min() >= 0
            assert tris.max() < len(verts)

    def test_qef_vs_average_similar_vertex_count(self):
        """QEF and average mode should produce the same topology."""
        field = _sphere_field()
        verts_qef, tris_qef, _ = extract_isosurface(field, isovalue=5.0, use_qef=True)
        verts_avg, tris_avg, _ = extract_isosurface(field, isovalue=5.0, use_qef=False)
        assert len(verts_qef) == len(verts_avg)
        assert len(tris_qef) == len(tris_avg)

    # ------------------------------------------------------------------
    # Normal vectors
    # ------------------------------------------------------------------

    def test_normals_unit_length(self):
        field = _sphere_field()
        _, _, normals = extract_isosurface(field, isovalue=5.0)
        if len(normals) > 0:
            lengths = np.linalg.norm(normals, axis=1)
            np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_non_3d_field_raises(self):
        with pytest.raises(ValueError):
            extract_isosurface(np.ones((10, 10)), isovalue=0.5)

    def test_float32_input_accepted(self):
        field = _sphere_field().astype(np.float32)
        verts, tris, normals = extract_isosurface(field, isovalue=5.0)
        assert verts.dtype == np.float64
