"""Cube topology constants for the Dual Marching Cubes algorithm.

Vertex and edge indexing follow the standard Marching Cubes convention:

    Vertex layout (bit encoding: bit-0 = i, bit-1 = j, bit-2 = k)::

        4----5
       /|   /|   k=1
      7----6 |
      | 0--|-1   k=0
      |/   |/
      3----2     j=1
     j=0

    Vertices:
        0: (0,0,0)   1: (1,0,0)   2: (1,1,0)   3: (0,1,0)
        4: (0,0,1)   5: (1,0,1)   6: (1,1,1)   7: (0,1,1)

    Edges (0-11):
        0-3  : four bottom-face edges  (k=0)
        4-7  : four top-face    edges  (k=1)
        8-11 : four vertical    edges
"""

import numpy as np

# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

#: (8, 3) int32 – vertex offsets (di, dj, dk) from cell origin
CUBE_VERTS = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ],
    dtype=np.int32,
)

#: (12, 2) int32 – pairs of vertex indices that form each cube edge
CUBE_EDGES = np.array(
    [
        [0, 1],  # 0  bottom x-edge  j=0, k=0
        [1, 2],  # 1  bottom y-edge  i=1, k=0
        [3, 2],  # 2  bottom x-edge  j=1, k=0
        [0, 3],  # 3  bottom y-edge  i=0, k=0
        [4, 5],  # 4  top x-edge     j=0, k=1
        [5, 6],  # 5  top y-edge     i=1, k=1
        [7, 6],  # 6  top x-edge     j=1, k=1
        [4, 7],  # 7  top y-edge     i=0, k=1
        [0, 4],  # 8  vertical z-edge i=0, j=0
        [1, 5],  # 9  vertical z-edge i=1, j=0
        [2, 6],  # 10 vertical z-edge i=1, j=1
        [3, 7],  # 11 vertical z-edge i=0, j=1
    ],
    dtype=np.int32,
)

# ---------------------------------------------------------------------------
# Adjacent-cell offsets for dual-mesh quad generation
# ---------------------------------------------------------------------------
# For each axis-aligned primal edge, four cells share that edge.
# The offsets below give the *two* free coordinates of the cell's corner
# relative to the edge's vertex with the smaller coordinate.
#
# Convention: CCW order when viewed from the POSITIVE axis direction.
# Winding is flipped in the core algorithm when the sign at v0 is "inside".

#: X-edge at (i, j, k)→(i+1, j, k).  Offsets (dj, dk) for the 4 cells.
X_EDGE_CELL_OFFSETS = np.array(
    [(0, 0), (-1, 0), (-1, -1), (0, -1)], dtype=np.int32
)

#: Y-edge at (i, j, k)→(i, j+1, k).  Offsets (di, dk) for the 4 cells.
Y_EDGE_CELL_OFFSETS = np.array(
    [(0, 0), (0, -1), (-1, -1), (-1, 0)], dtype=np.int32
)

#: Z-edge at (i, j, k)→(i, j, k+1).  Offsets (di, dj) for the 4 cells.
Z_EDGE_CELL_OFFSETS = np.array(
    [(0, 0), (-1, 0), (-1, -1), (0, -1)], dtype=np.int32
)
