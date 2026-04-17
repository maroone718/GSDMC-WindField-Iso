# GSDMC Wind Speed Isosurface Extraction

This repository contains a Python implementation of GSDMC (Gradient-Similarity Driven Marching Cubes), 
a method for extracting wind-speed isosurfaces from three-dimensional NetCDF wind-field data.

The main workflow is:

1. read a 3D wind-speed field from an `.nc` file;
2. compute gradient and local similarity fields;
3. classify grid edges into flat and non-flat regions;
4. place isosurface vertices with midpoint or Hermite interpolation;
5. assemble triangles with Marching Cubes lookup tables;
6. export OBJ meshes and visualize the result with PyVista.

## Repository Structure

```text
.
|-- GSDMC.py              # Core GSDMC algorithm and NetCDF loader
|-- math_utils.py         # Gradient, similarity, and interpolation utilities
|-- mc_tables.py          # Marching Cubes lookup tables
|-- run_gsdmc.py   # Example script for wind-field extraction
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation
```

Generated folders such as `outputs/` and `__pycache__/` are not required for
the source release. NetCDF data files are not included by default; users should
prepare compatible input data before running the example script.

## Algorithm Overview

GSDMC extends the conventional Marching Cubes framework by combining unified edge-topology lookup, 
vertex reuse, and gradient-similarity driven adaptive intersection computation.

For each edge, the algorithm uses:

- gradient magnitude `G`;
- local similarity `S`;
- adaptive thresholds `T_G` and `T_S`.

The edge classification rule is:

```text
if average(G) <= T_G and average(S) >= T_S:
    flat region -> midpoint interpolation
else:
    non-flat region -> Hermite interpolation
```

This is implemented in `GSDMC._is_flat_edge()` and
`GSDMC._interpolate_vertex()`.

## Installation

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `scipy`
- `netCDF4`
- `pyvista`
- `numba`

## Input Data Format

Users need to prepare a NetCDF file with the same data layout expected by the
loader. The default example script is configured to read:

- wind-speed variable: `wind_Prediction_0`
- coordinate variables: `x`, `y`, `z`

The loader also attempts to detect common coordinate names such as
`lon`/`longitude`, `lat`/`latitude`, and `height`/`altitude`.

Place the file in the project directory as `sample_data.nc`, or edit `nc_file`
in `run_gsdmc.py` to point to your data file.

Missing values should be handled before extraction to avoid NaN propagation in gradient and similarity computations. 
In the current implementation, missing values are filled with 0.0, so users should ensure that this treatment is appropriate for their data.

## Quick Start

1. Prepare a compatible NetCDF file. The repository does not include sample NetCDF data, 
  so users should provide their own input file. Place it in the project directory as sample_data.nc, or update nc_file in the driver script.

2. Check the configuration block:

```python
nc_file = "./sample_data.nc"
variable_name = "wind_Prediction_0"
sigma = 0.4
```

3. Run:

```bash
python run_gsdmc.py
```

The script will:

- load the wind field;
- print data statistics;
- ask for isosurface thresholds;
- extract one or more isosurfaces;
- optionally export OBJ files;
- open a PyVista visualization window.

## Outputs

By default, generated files are written to:

```text
outputs/obj      # exported OBJ meshes
outputs/images   # optional screenshots from PyVista visualization
```

You can override these paths with environment variables:

```powershell
$env:GSDMC_OBJ_DIR = "D:\path\to\obj"
$env:GSDMC_IMAGE_DIR = "D:\path\to\images"
```

OBJ files use the standard format:

```text
v x y z
f i j k
```

Face indices are 1-based, as required by the OBJ specification.

## Notes

- The example script uses `n_jobs=1` by default for memory safety.
- The Z direction is scaled by `50x` in vertex coordinates to make low vertical
  wind-field layers easier to inspect visually.
- NetCDF input files are not included in this repository. 
  Users should provide their own compatible data files in the format described above.
- This repository provides the implementation of the proposed GSDMC method only. 
  Baseline and ablation variants discussed in the paper are not included in this source release.

## Minimal Usage From Python

```python
from GSDMC import GSDMC, load_wind_nc

field, spacing, metadata = load_wind_nc("./sample_data.nc", "wind_Prediction_0")

extractor = GSDMC(
    field,
    spacing=spacing,
    sigma=0.4,
    verbose=True,
    n_jobs=1,
)

vertices, triangles = extractor.extract_isosurface(isovalue=8.0)
extractor.export_obj(vertices, triangles, "wind_8.0ms.obj")
```

## Suggested Citation

If you use this code in an academic project, please cite the corresponding
paper or technical report describing the GSDMC method.
