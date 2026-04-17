# GSDMC-WindField-Iso

Python implementation of **Generalized Symmetric Dual Marching Cubes (GSDMC)**
for wind-speed isosurface extraction from 3-D NetCDF wind-field data.

---

## Overview

GSDMC is a dual-contouring algorithm that extracts isosurfaces from scalar
fields defined on regular 3-D grids.  Unlike traditional Marching Cubes,
which places vertices *on cell edges*, the Dual Marching Cubes approach places
a single vertex *inside each active cell* using **Quadratic Error Function
(QEF)** minimisation.  This produces:

- sharper feature preservation at creases and corners,
- symmetric meshes (swapping inside/outside yields the same mesh, just flipped), and
- no multi-case look-up tables required.

This library applies GSDMC to wind-field data stored in NetCDF files
(ERA5, WRF, ECMWF, etc.) to extract constant-wind-speed isosurfaces (speed
shells).

---

## Quick start

```python
from gsdmc import read_wind_field, extract_isosurface, Mesh

# 1. Read wind-field data from a NetCDF file
data = read_wind_field("era5_wind.nc")
print(data)   # WindFieldData(shape=(37, 721, 1440), speed=[0.01, 32.4] m/s)

# 2. Extract the 15 m/s isosurface
vertices, triangles, normals = extract_isosurface(
    data.speed,
    isovalue=15.0,
    spacing=data.spacing,
    origin=data.origin,
)
print(f"Extracted {len(triangles)} triangles")

# 3. Save the mesh
mesh = Mesh(vertices, triangles, normals)
mesh.save_obj("wind_15ms.obj")    # Wavefront OBJ
mesh.save_ply("wind_15ms.ply")    # PLY (for MeshLab / Blender)
mesh.save_stl("wind_15ms.stl")    # STL (for 3-D printing)
```

### Choosing an isovalue

```python
from gsdmc import read_wind_field, wind_speed_percentile

data = read_wind_field("era5_wind.nc")

# Use the 90th-percentile wind speed as the isovalue
isovalue = wind_speed_percentile(data.speed, 90)
print(f"90th-percentile wind speed: {isovalue:.2f} m/s")
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

**Dependencies**

| Package  | Version  | Purpose                           |
|----------|----------|-----------------------------------|
| numpy    | >= 1.20  | Array operations                  |
| scipy    | >= 1.6   | Linear algebra (QEF solver)       |
| netCDF4  | >= 1.5   | Reading NetCDF wind-field files   |

---

## Algorithm

### Generalized Symmetric Dual Marching Cubes

1. **Classification** - each grid vertex is labelled *inside*
   (`wind_speed >= isovalue`) or *outside*.

2. **Active cells** - a cell (cube) is *active* when its 8 corners are not
   all the same label.  The Marching Cubes case index (0-255) identifies
   the topological configuration.

3. **QEF vertex placement** - for each active cell, the intersection points
   on its 12 edges are computed by linear interpolation.  The gradient of the
   scalar field (finite-difference approximation) provides surface normals at
   these points.  A QEF is solved to find the cell's optimal dual vertex:

   > minimise  sum_i ( n_i . (x - p_i) )^2

   The solution is regularised toward the mass point (mean of intersection
   points) to handle planar / edge-feature cells robustly.

4. **Dual mesh** - for every *active primal edge* (an edge whose two
   endpoints have opposite labels), the four cells sharing that edge each
   contribute one dual vertex.  These four vertices form a quad, split into
   two triangles.  Winding order is chosen so that face normals point
   outward (away from the high-wind region).

### NetCDF reader

The reader supports:

- **ERA5** style - variables `u`, `v`, `w` (or `u10`, `v10`); dimensions
  `(time, level, latitude, longitude)`.
- **WRF** style - variables `U`, `V`, `W`; dimensions
  `(Time, bottom_top, south_north, west_east)`.
- **Generic** - caller specifies variable names explicitly.

---

## API reference

### `gsdmc.extract_isosurface`

```python
vertices, triangles, vertex_normals = extract_isosurface(
    scalar_field,           # ndarray (nx, ny, nz)
    isovalue,               # float
    spacing=(1, 1, 1),      # (dx, dy, dz) physical cell size
    origin=(0, 0, 0),       # physical coords of grid point (0,0,0)
    use_qef=True,           # False -> faster average-point placement
)
```

### `gsdmc.read_wind_field`

```python
data = read_wind_field(
    path,                   # str | Path to NetCDF file
    u_var=None,             # override U variable name
    v_var=None,             # override V variable name
    w_var=None,             # override W variable name
    time_idx=0,             # which time step to load
    fill_value=0.0,         # replacement for masked cells
)
# data.speed   - (nz, ny, nx) float64 wind-speed array
# data.spacing - (dz, dy, dx) grid spacing
# data.origin  - (z0, y0, x0) physical origin
```

### `gsdmc.Mesh`

```python
mesh = Mesh(vertices, triangles, vertex_normals=None)
mesh.save_obj(path)   # Wavefront OBJ
mesh.save_ply(path)   # ASCII PLY
mesh.save_stl(path)   # ASCII STL
mesh = Mesh.load_obj(path)
```

### `gsdmc.wind_speed_percentile`

```python
isovalue = wind_speed_percentile(speed_array, percentile=90)
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT (c) 2026 Zhao ZhongYi