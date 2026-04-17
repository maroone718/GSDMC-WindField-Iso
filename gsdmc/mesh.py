"""Mesh data structure and file I/O for isosurface output.

Supported output formats
------------------------
* **OBJ**  – Wavefront Object, widely supported in 3-D tools.
* **PLY**  – Polygon File Format (ASCII), suitable for MeshLab, Blender, etc.
* **STL**  – Stereolithography (ASCII), useful for 3-D printing pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Mesh:
    """A triangulated surface mesh.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Vertex coordinates.
    triangles : ndarray, shape (M, 3), dtype int
        Triangle faces given as vertex-index triplets.
    vertex_normals : ndarray, shape (N, 3), optional
        Per-vertex outward unit normals.  Recomputed from face normals when
        not provided.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        vertex_normals: np.ndarray | None = None,
    ):
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float64)
        self.triangles: np.ndarray = np.asarray(triangles, dtype=np.int32)
        if vertex_normals is not None:
            self.vertex_normals: np.ndarray = np.asarray(vertex_normals, dtype=np.float64)
        else:
            self.vertex_normals = self._compute_vertex_normals()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Mesh(vertices={self.n_vertices}, triangles={self.n_triangles})"
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _compute_vertex_normals(self) -> np.ndarray:
        """Compute per-vertex normals by averaging adjacent face normals."""
        vn = np.zeros_like(self.vertices)
        if self.n_triangles == 0:
            return vn
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)  # (M, 3)
        # Accumulate (weighted by area – length of cross product)
        for i, tri in enumerate(self.triangles):
            vn[tri[0]] += face_normals[i]
            vn[tri[1]] += face_normals[i]
            vn[tri[2]] += face_normals[i]
        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return vn / norms

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(min_corner, max_corner)`` of the mesh's AABB."""
        if self.n_vertices == 0:
            z = np.zeros(3)
            return z, z
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    # ------------------------------------------------------------------
    # I/O – OBJ
    # ------------------------------------------------------------------

    def save_obj(self, path: str | Path) -> None:
        """Write the mesh to a Wavefront OBJ file.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in ``.obj``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write("# GSDMC-WindField-Iso output\n")
            for x, y, z in self.vertices:
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for nx, ny, nz in self.vertex_normals:
                fh.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for a, b, c in self.triangles:
                # OBJ indices are 1-based
                fh.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n")

    @classmethod
    def load_obj(cls, path: str | Path) -> "Mesh":
        """Load a Wavefront OBJ file (vertices and faces only).

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Mesh
        """
        path = Path(path)
        vertices: list[list[float]] = []
        triangles: list[list[int]] = []
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    # handle "v//vn", "v/vt/vn", "v/vt", "v" formats
                    face = [int(p.split("/")[0]) - 1 for p in parts]
                    if len(face) == 3:
                        triangles.append(face)
                    elif len(face) == 4:
                        # quad → two triangles
                        triangles.append([face[0], face[1], face[2]])
                        triangles.append([face[0], face[2], face[3]])
        return cls(np.array(vertices), np.array(triangles, dtype=np.int32))

    # ------------------------------------------------------------------
    # I/O – PLY (ASCII)
    # ------------------------------------------------------------------

    def save_ply(self, path: str | Path) -> None:
        """Write the mesh to an ASCII PLY file.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nv = self.n_vertices
        nt = self.n_triangles
        with path.open("w") as fh:
            fh.write(
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {nv}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                f"element face {nt}\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
            )
            for (x, y, z), (nx, ny, nz) in zip(self.vertices, self.vertex_normals):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for a, b, c in self.triangles:
                fh.write(f"3 {a} {b} {c}\n")

    # ------------------------------------------------------------------
    # I/O – STL (ASCII)
    # ------------------------------------------------------------------

    def save_stl(self, path: str | Path) -> None:
        """Write the mesh to an ASCII STL file.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write("solid gsdmc_isosurface\n")
            for tri, fn in zip(self.triangles, self._face_normals()):
                fh.write(f"  facet normal {fn[0]:.6f} {fn[1]:.6f} {fn[2]:.6f}\n")
                fh.write("    outer loop\n")
                for vi in tri:
                    x, y, z = self.vertices[vi]
                    fh.write(f"      vertex {x:.6f} {y:.6f} {z:.6f}\n")
                fh.write("    endloop\n")
                fh.write("  endfacet\n")
            fh.write("endsolid gsdmc_isosurface\n")

    def _face_normals(self) -> np.ndarray:
        """Compute and return unit face normals, shape (M, 3)."""
        if self.n_triangles == 0:
            return np.zeros((0, 3))
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(fn, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return fn / norms
