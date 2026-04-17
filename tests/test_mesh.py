"""Tests for gsdmc.mesh – Mesh data structure and I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gsdmc.mesh import Mesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _unit_tetrahedron() -> Mesh:
    """Minimal valid mesh: a tetrahedron (4 vertices, 4 triangular faces)."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return Mesh(verts, tris)


def _empty_mesh() -> Mesh:
    return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------

class TestMeshProperties:
    def test_n_vertices(self):
        m = _unit_tetrahedron()
        assert m.n_vertices == 4

    def test_n_triangles(self):
        m = _unit_tetrahedron()
        assert m.n_triangles == 4

    def test_vertex_normals_unit_length(self):
        m = _unit_tetrahedron()
        lengths = np.linalg.norm(m.vertex_normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    def test_provided_normals_stored(self):
        m = _unit_tetrahedron()
        provided = np.tile([0.0, 0.0, 1.0], (4, 1))
        m2 = Mesh(m.vertices, m.triangles, vertex_normals=provided)
        np.testing.assert_array_equal(m2.vertex_normals, provided)

    def test_bounding_box(self):
        m = _unit_tetrahedron()
        lo, hi = m.bounding_box()
        np.testing.assert_array_equal(lo, [0, 0, 0])
        np.testing.assert_array_equal(hi, [1, 1, 1])

    def test_empty_mesh_bounding_box(self):
        m = _empty_mesh()
        lo, hi = m.bounding_box()
        np.testing.assert_array_equal(lo, [0, 0, 0])

    def test_vertex_dtype(self):
        verts = np.ones((3, 3), dtype=np.float32)
        tris = np.array([[0, 1, 2]])
        m = Mesh(verts, tris)
        assert m.vertices.dtype == np.float64

    def test_triangle_dtype(self):
        verts = np.eye(3, dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        m = Mesh(verts, tris)
        assert m.triangles.dtype == np.int32


# ---------------------------------------------------------------------------
# OBJ I/O
# ---------------------------------------------------------------------------

class TestOBJIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.obj"
        m.save_obj(path)
        m2 = Mesh.load_obj(path)
        np.testing.assert_allclose(m2.vertices, m.vertices, atol=1e-5)
        # face count may differ if OBJ loader splits quads, but tets are all tris
        assert m2.n_triangles == m.n_triangles

    def test_obj_file_created(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "sub" / "tet.obj"
        m.save_obj(path)
        assert path.exists()

    def test_obj_header_present(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.obj"
        m.save_obj(path)
        content = path.read_text()
        assert "v " in content
        assert "f " in content

    def test_obj_vertex_normal_lines(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.obj"
        m.save_obj(path)
        content = path.read_text()
        assert "vn " in content

    def test_obj_empty_mesh(self, tmp_path):
        m = _empty_mesh()
        path = tmp_path / "empty.obj"
        m.save_obj(path)  # should not raise

    def test_load_quad_faces_split(self, tmp_path):
        """OBJ loader should split quad faces into two triangles."""
        path = tmp_path / "quad.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
        )
        m = Mesh.load_obj(path)
        assert m.n_triangles == 2


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

class TestPLYIO:
    def test_ply_file_created(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.ply"
        m.save_ply(path)
        assert path.exists()

    def test_ply_header(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.ply"
        m.save_ply(path)
        content = path.read_text()
        assert "element vertex 4" in content
        assert "element face 4" in content

    def test_ply_empty_mesh(self, tmp_path):
        m = _empty_mesh()
        path = tmp_path / "empty.ply"
        m.save_ply(path)

    def test_ply_vertex_count_matches(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.ply"
        m.save_ply(path)
        lines = path.read_text().splitlines()
        # Count data lines after 'end_header'
        data_start = next(i for i, l in enumerate(lines) if l == "end_header") + 1
        vertex_lines = lines[data_start : data_start + m.n_vertices]
        assert len(vertex_lines) == 4  # 4 vertices, each with 6 floats


# ---------------------------------------------------------------------------
# STL I/O
# ---------------------------------------------------------------------------

class TestSTLIO:
    def test_stl_file_created(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.stl"
        m.save_stl(path)
        assert path.exists()

    def test_stl_solid_keyword(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.stl"
        m.save_stl(path)
        content = path.read_text()
        assert content.startswith("solid")
        assert "endsolid" in content

    def test_stl_facet_count(self, tmp_path):
        m = _unit_tetrahedron()
        path = tmp_path / "tet.stl"
        m.save_stl(path)
        content = path.read_text()
        assert content.count("facet normal") == m.n_triangles

    def test_stl_empty_mesh(self, tmp_path):
        m = _empty_mesh()
        path = tmp_path / "empty.stl"
        m.save_stl(path)
