"""GSDMC-WindField-Iso – Python implementation of Generalized Symmetric Dual
Marching Cubes for wind-speed isosurface extraction from 3D NetCDF data.

Quick-start example::

    from gsdmc import read_wind_field, extract_isosurface, Mesh

    data = read_wind_field("era5_wind.nc")
    vertices, triangles, normals = extract_isosurface(data.speed, isovalue=15.0,
                                                      spacing=data.spacing,
                                                      origin=data.origin)
    mesh = Mesh(vertices, triangles, normals)
    mesh.save_obj("wind_isosurface.obj")
"""

from .core import extract_isosurface
from .mesh import Mesh
from .reader import WindFieldData, read_wind_field
from .wind import compute_wind_speed, wind_speed_percentile

__version__ = "0.1.0"

__all__ = [
    "extract_isosurface",
    "read_wind_field",
    "WindFieldData",
    "compute_wind_speed",
    "wind_speed_percentile",
    "Mesh",
]
