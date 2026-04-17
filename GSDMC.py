"""
GSDMC (Gradient and Similarity Driven Marching Cubes) Algorithm
"""

import numpy as np
import netCDF4 as nc
import pyvista as pv
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from mc_tables import EDGE_TABLE, TRIANGLE_TABLE, EDGE_VERTICES, CUBE_VERTICES
from math_utils import (
    compute_gradient_field,
    compute_similarity_field,
    hermite_interpolation,
    linear_interpolation,
    midpoint_interpolation
)


class GSDMC:
    VALID_MODES = ("MC", "MC_ULVR", "GSDMC")

    def __init__(self, field, spacing, sigma=0.4, verbose=True, n_jobs=-1, mode="GSDMC"):
        
        self.field = field
        self.spacing = np.array(spacing, dtype=np.float32)
        self.shape = field.shape
        self.nz, self.ny, self.nx = self.shape
        self.verbose = verbose
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self.mode = mode.upper()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of {self.VALID_MODES}")
        self.use_vertex_reuse = self.mode in ("MC_ULVR", "GSDMC")
        
        if self.verbose and self.n_jobs > 1:
            print(f"Multiprocessing enabled: {self.n_jobs} worker processes")
        
        # Preprocessing phase
        if self.verbose:
            print("=" * 50)
            print("GSDMC Initialization")
            print(f"Mode: {self.mode}")
            print(f"Data dimensions: {self.shape}")
            print(f"Physical spacing: {spacing}")
            total_voxels = np.prod(self.shape)
            memory_mb = total_voxels * 4 * 3 / (1024**2)  # 3 float32 arrays
            print(f"Data points: {total_voxels:,}")
            print(f"Estimated memory: ~{memory_mb:.0f}MB")
            print("=" * 50)
        
        total_start = time.time()
        
        self.G = None
        self.grads = None
        self.S = None
        self.T_G = None
        self.T_S = None

        if self.mode == "GSDMC":
            # 1. Compute gradient field
            if self.verbose:
                print("Step 1: Computing gradient field...")
            t1 = time.time()
            self.G, self.grads = compute_gradient_field(field, spacing)
            t1_elapsed = time.time() - t1
            if self.verbose:
                print(f"  Complete (time: {t1_elapsed:.2f}s)")
            
            # 2. Compute similarity field
            if self.verbose:
                print("Step 2: Computing similarity field (Numba JIT accelerated)...")
            t2 = time.time()
            self.S = compute_similarity_field(field, sigma)
            t2_elapsed = time.time() - t2
            if self.verbose:
                print(f"  Complete (time: {t2_elapsed:.2f}s)")
                if t2_elapsed < 5.0:
                    print(f"Numba JIT compilation cached, next run will be faster!")
                speedup = 60.0 / max(0.1, t2_elapsed)  # Original version ~60s
                if speedup > 5:
                    print(f"Speedup ~{speedup:.0f}x compared to original version")
            
            # 3. Compute adaptive thresholds
            if self.verbose:
                print("Step 3: Computing adaptive thresholds...")
            t3 = time.time()
            self.T_G = np.percentile(self.G, 75)
            self.T_S = np.median(self.S)
            t3_elapsed = time.time() - t3
        else:
            t3_elapsed = 0.0
            if self.verbose:
                print("Skipping gradient/similarity preprocessing for standard MC interpolation mode.")
        
        total_elapsed = time.time() - total_start
        
        if self.verbose:
            if self.mode == "GSDMC":
                print(f"  Complete (time: {t3_elapsed:.2f}s)")
                print(f"  - Gradient threshold T_G = {self.T_G:.4f}")
                print(f"  - Similarity threshold T_S = {self.T_S:.4f}")
            print("=" * 50)
            print(f"Total preprocessing time: {total_elapsed:.2f}s")
            print("=" * 50)
        
        # Vertex cache and result containers
        self.vertex_cache = {}
        self.vertices = []
        self.triangles = []
    
    def _get_node_id(self, i, j, k):
        """Calculate globally unique index for a node"""
        return k * (self.ny * self.nx) + j * self.nx + i
    
    def _get_edge_key(self, i1, j1, k1, i2, j2, k2):
        """Generate unique identifier for an edge (for caching)"""
        node_a = self._get_node_id(i1, j1, k1)
        node_b = self._get_node_id(i2, j2, k2)
        return (min(node_a, node_b), max(node_a, node_b))
    
    def _is_flat_edge(self, i1, j1, k1, i2, j2, k2):
        """
        Determine if edge is in flat region
        
        Decision logic (paper core):
            IF (G_avg <= T_G) AND (S_avg >= T_S):
                -> Flat region
            ELSE:
                -> Non-flat region
        """
        G1 = self.G[k1, j1, i1]
        G2 = self.G[k2, j2, i2]
        S1 = self.S[k1, j1, i1]
        S2 = self.S[k2, j2, i2]
        
        G_avg = (G1 + G2) / 2.0
        S_avg = (S1 + S2) / 2.0
        
        return (G_avg <= self.T_G) and (S_avg >= self.T_S)
    
    def _interpolate_vertex(self, i1, j1, k1, i2, j2, k2, iso):
        """
        Interpolate vertex on edge
        
        Parameters:
            (i1,j1,k1), (i2,j2,k2): Grid coordinates of two edge endpoints
            iso: Isovalue threshold
        
        Returns:
            Physical coordinates of interpolated point [x, y, z]
        """
        # Calculate physical coordinates (Z direction scaled 50x)
        spacing_with_exaggeration = np.array([self.spacing[0], self.spacing[1], self.spacing[2] * 50], dtype=np.float32)
        p1 = np.array([i1, j1, k1], dtype=np.float32) * spacing_with_exaggeration
        p2 = np.array([i2, j2, k2], dtype=np.float32) * spacing_with_exaggeration
        
        # Get endpoint scalar values
        v1 = self.field[k1, j1, i1]
        v2 = self.field[k2, j2, i2]
        
        if self.mode in ("MC", "MC_ULVR"):
            return linear_interpolation(p1, p2, v1, v2, iso)

        # Determine interpolation method
        if self._is_flat_edge(i1, j1, k1, i2, j2, k2):
            # Flat region: use midpoint method
            return midpoint_interpolation(p1, p2)
        else:
            # Non-flat region: use Hermite interpolation
            grad1 = np.array([
                self.grads[0][k1, j1, i1],  # grad_x
                self.grads[1][k1, j1, i1],  # grad_y
                self.grads[2][k1, j1, i1]   # grad_z
            ], dtype=np.float32)
            
            grad2 = np.array([
                self.grads[0][k2, j2, i2],
                self.grads[1][k2, j2, i2],
                self.grads[2][k2, j2, i2]
            ], dtype=np.float32)
            
            return hermite_interpolation(p1, p2, v1, v2, grad1, grad2, iso)
    
    def _process_cube(self, i, j, k, iso):
        """
        Process a single voxel cube
        
        Parameters:
            i, j, k: Lower-left corner coordinates of voxel
            iso: Isovalue threshold
        """
        # 1. Get scalar values at 8 vertices
        cube_values = np.zeros(8, dtype=np.float32)
        for vi, (di, dj, dk) in enumerate(CUBE_VERTICES):
            ii, jj, kk = i + di, j + dj, k + dk
            if ii < self.nx and jj < self.ny and kk < self.nz:
                cube_values[vi] = self.field[kk, jj, ii]
        
        # 2. Calculate Cube Index (8-bit binary code)
        cube_idx = 0
        for vi in range(8):
            if cube_values[vi] < iso:
                cube_idx |= (1 << vi)
        
        # 3. Boundary cases: no surface or entirely within surface
        if cube_idx == 0 or cube_idx == 255:
            return
        
        # 4. Find intersecting edges
        edge_flags = EDGE_TABLE[cube_idx]
        if edge_flags == 0:
            return
        
        # 5. Process each intersecting edge (maximum 12)
        vert_indices = [-1] * 12
        
        for edge_id in range(12):
            if not (edge_flags & (1 << edge_id)):
                continue
            
            # Get two endpoints of the edge
            v1_local, v2_local = EDGE_VERTICES[edge_id]
            
            di1, dj1, dk1 = CUBE_VERTICES[v1_local]
            di2, dj2, dk2 = CUBE_VERTICES[v2_local]
            
            i1, j1, k1 = i + di1, j + dj1, k + dk1
            i2, j2, k2 = i + di2, j + dj2, k + dk2
            
            if self.use_vertex_reuse:
                # Generate unique edge key
                edge_key = self._get_edge_key(i1, j1, k1, i2, j2, k2)
                
                # Check cache
                if edge_key in self.vertex_cache:
                    vert_indices[edge_id] = self.vertex_cache[edge_key]
                else:
                    # Calculate new vertex
                    vertex_pos = self._interpolate_vertex(i1, j1, k1, i2, j2, k2, iso)
                    
                    # Add to vertex list
                    new_id = len(self.vertices)
                    self.vertices.append(vertex_pos)
                    
                    # Store in cache
                    self.vertex_cache[edge_key] = new_id
                    vert_indices[edge_id] = new_id
            else:
                vertex_pos = self._interpolate_vertex(i1, j1, k1, i2, j2, k2, iso)
                new_id = len(self.vertices)
                self.vertices.append(vertex_pos)
                vert_indices[edge_id] = new_id
        
        # 6. Generate triangles
        tri_config = TRIANGLE_TABLE[cube_idx]
        for i in range(0, len(tri_config), 3):
            if i + 2 < len(tri_config):
                triangle = [
                    vert_indices[tri_config[i]],
                    vert_indices[tri_config[i+1]],
                    vert_indices[tri_config[i+2]]
                ]
                self.triangles.append(triangle)
    
    def extract_isosurface(self, isovalue):
       
        if self.verbose:
            print(f"Starting isosurface extraction (isovalue={isovalue})...")
        
        extract_start = time.time()
        
        # Clear cache
        self.vertex_cache.clear()
        self.vertices = []
        self.triangles = []
        
        # Sparse optimization: pre-mark active voxels
        if self.verbose:
            print("Phase 1: Marking active voxels (sparse optimization)...")
        t_sparse = time.time()
        active_cubes = self._mark_active_cubes(isovalue)
        n_active = np.sum(active_cubes)
        total_cubes = (self.nx - 1) * (self.ny - 1) * (self.nz - 1)
        sparse_ratio = n_active / total_cubes
        t_sparse_elapsed = time.time() - t_sparse
        
        if self.verbose:
            print(f"    Active voxels: {n_active:,} / {total_cubes:,} ({sparse_ratio*100:.1f}%)")
            print(f"    Time: {t_sparse_elapsed:.2f}s")
            
        # Choose processing method based on n_jobs and memory safety
        # Calculate data size to avoid memory overflow
        data_size_bytes = self.field.nbytes
        if self.mode == "GSDMC":
            data_size_bytes += self.G.nbytes + self.S.nbytes + self.grads[0].nbytes * 3
        data_size_mb = data_size_bytes / (1024**2)
        safe_for_multiprocessing = (self.mode == "GSDMC" and data_size_mb < 500 and n_active > 5000)
        
        if self.n_jobs > 1 and safe_for_multiprocessing:
            # Multiprocessing parallel
            if self.verbose:
                print(f"Phase 2: Multiprocessing ({min(self.n_jobs, 8)} processes)...")
                print(f"    Data size: {data_size_mb:.1f}MB")
            t_process = time.time()
            try:
                vertices, triangles = self._parallel_process(active_cubes, isovalue)
                t_process_elapsed = time.time() - t_process
                processed = n_active
            except (MemoryError, Exception) as e:
                if self.verbose:
                    print(f"Multiprocessing failed, fallback to serial: {type(e).__name__}")
                # Fallback to serial processing
                t_process = time.time()
                processed = 0
                last_progress_time = time.time()
                
                active_indices = np.argwhere(active_cubes)
                for idx, (k, j, i) in enumerate(active_indices):
                    self._process_cube(i, j, k, isovalue)
                    processed += 1
                    
                    if self.verbose and (time.time() - last_progress_time > 0.5 or idx == len(active_indices) - 1):
                        progress = (processed / n_active) * 100
                        elapsed = time.time() - t_process
                        speed = processed / elapsed if elapsed > 0 else 0
                        eta = (n_active - processed) / speed if speed > 0 else 0
                        print(f"    Progress: {progress:.1f}% ({processed:,}/{n_active:,}) | "
                              f"Speed: {speed:.0f} cube/s | ETA: {eta:.1f}s")
                        last_progress_time = time.time()
                
                t_process_elapsed = time.time() - t_process
                vertices = np.array(self.vertices, dtype=np.float32)
                triangles = np.array(self.triangles, dtype=np.int32)
        else:
            # Serial processing (few voxels, parallel disabled, or data too large)
            if self.verbose:
                reason = "data too large" if data_size_mb >= 500 else "few voxels" if n_active <= 5000 else "parallel disabled"
                print(f"Phase 2: Serial processing active voxels ({reason})...")
            t_process = time.time()
            processed = 0
            last_progress_time = time.time()
            
            active_indices = np.argwhere(active_cubes)
            for idx, (k, j, i) in enumerate(active_indices):
                self._process_cube(i, j, k, isovalue)
                processed += 1
                
                # Update progress every 0.5 seconds
                if self.verbose and (time.time() - last_progress_time > 0.5 or idx == len(active_indices) - 1):
                    progress = (processed / n_active) * 100
                    elapsed = time.time() - t_process
                    speed = processed / elapsed if elapsed > 0 else 0
                    eta = (n_active - processed) / speed if speed > 0 else 0
                    print(f"    Progress: {progress:.1f}% ({processed:,}/{n_active:,}) | "
                          f"Speed: {speed:.0f} cube/s | ETA: {eta:.1f}s")
                    last_progress_time = time.time()
            
            t_process_elapsed = time.time() - t_process
            
            # Convert to Numpy arrays
            vertices = np.array(self.vertices, dtype=np.float32)
            triangles = np.array(self.triangles, dtype=np.int32)
        
        extract_total = time.time() - extract_start
        
        if self.verbose:
            print("=" * 50)
            print("Extraction complete!")
            print(f"Statistics:")
            print(f"  - Vertices: {len(vertices):,}")
            print(f"  - Triangles: {len(triangles):,}")
            if self.use_vertex_reuse:
                print(f"  - Vertex reuse rate: {1 - len(vertices)/max(1, processed*12):.2%}")
            else:
                print("  - Vertex reuse: disabled")
            print(f"  - Sparse optimization savings: {(1-sparse_ratio)*100:.1f}% of computation")
            print(f"Time breakdown:")
            print(f"  - Mark active voxels: {t_sparse_elapsed:.2f}s")
            print(f"  - Process voxels: {t_process_elapsed:.2f}s")
            print(f"  - Total: {extract_total:.2f}s")
            print(f"  - Average speed: {processed/max(0.001, t_process_elapsed):.0f} cube/s")
            print("=" * 50)
        
        return vertices, triangles
    
    def _mark_active_cubes(self, isovalue):
        """
        Mark active voxels containing isosurface (vectorized optimization)
        
        Parameters:
            isovalue: Isovalue threshold
        
        Returns:
            active_cubes: Boolean array (nz-1, ny-1, nx-1)
        """
        # Vectorized version: process all cubes at once
        # Get values at 8 vertices
        v000 = self.field[:-1, :-1, :-1]
        v100 = self.field[:-1, :-1, 1:]
        v010 = self.field[:-1, 1:, :-1]
        v110 = self.field[:-1, 1:, 1:]
        v001 = self.field[1:, :-1, :-1]
        v101 = self.field[1:, :-1, 1:]
        v011 = self.field[1:, 1:, :-1]
        v111 = self.field[1:, 1:, 1:]
        
        # Calculate min and max values for each cube
        min_val = np.minimum.reduce([v000, v100, v010, v110, v001, v101, v011, v111])
        max_val = np.maximum.reduce([v000, v100, v010, v110, v001, v101, v011, v111])
        
        # Check if isovalue is within range
        active_cubes = (min_val <= isovalue) & (isovalue <= max_val)
        
        return active_cubes
    
    def _parallel_process(self, active_cubes, isovalue):
        """
        Multiprocessing parallel processing of active voxels (memory-optimized version)
        
        Parameters:
            active_cubes: Active voxels boolean array
            isovalue: Isovalue threshold
        
        Returns:
            vertices: Vertex array
            triangles: Triangle array
        """
        # Get indices of all active voxels
        active_indices = np.argwhere(active_cubes)
        
        # Limit process count to avoid memory overflow
        n_active = len(active_indices)
        actual_jobs = min(self.n_jobs, 8, max(2, n_active // 2000))  # Max 8 processes
        
        # Use smaller chunks to reduce single data transfer size
        chunk_size = max(200, n_active // (actual_jobs * 2))  # Smaller chunks
        chunks = [active_indices[i:i+chunk_size] for i in range(0, n_active, chunk_size)]
        
        # Create process pool for parallel processing (using actual process count)
        process_func = partial(_process_chunk_worker, 
                               field=self.field,
                               G=self.G,
                               S=self.S,
                               grads=self.grads,
                               spacing=self.spacing,
                               T_G=self.T_G,
                               T_S=self.T_S,
                               isovalue=isovalue)
        
        with Pool(processes=actual_jobs) as pool:
            results = pool.map(process_func, chunks, chunksize=1)
        
        # Merge results
        all_vertices = []
        all_triangles = []
        vertex_offset = 0
        
        for chunk_vertices, chunk_triangles in results:
            if len(chunk_vertices) > 0:
                all_vertices.append(chunk_vertices)
                # Adjust triangle indices
                all_triangles.append(chunk_triangles + vertex_offset)
                vertex_offset += len(chunk_vertices)
        
        # Merge into final arrays
        if all_vertices:
            vertices = np.vstack(all_vertices)
            triangles = np.vstack(all_triangles)
        else:
            vertices = np.array([], dtype=np.float32).reshape(0, 3)
            triangles = np.array([], dtype=np.int32).reshape(0, 3)
        
        return vertices, triangles
    
    def export_obj(self, vertices, triangles, filename):
        
        with open(filename, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write faces (OBJ indices start from 1)
            for tri in triangles:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        
        if self.verbose:
            print(f"Exported to: {filename}")
    
    def visualize_with_pyvista(self, vertices, triangles, title="GSDMC Isosurface", 
                               show_edges=True, color='lightblue', opacity=1.0):
        
        # Create PyVista mesh
        # Need to convert triangles to PyVista format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).flatten()
        
        mesh = pv.PolyData(vertices, faces)
        
        # Add height exaggeration note
        if self.verbose:
            print(f"Note: Z direction scaled 50x to enhance visualization")
        
        # Calculate mesh statistics
        mesh.compute_normals(inplace=True)
        
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color=color, show_edges=show_edges, 
                        opacity=opacity, smooth_shading=True)
        
        # Add coordinate axes
        plotter.add_axes()
        plotter.add_bounding_box()
        
        # Set title
        plotter.add_text(title, font_size=12, position='upper_left')
        
        # Display mesh info (including height exaggeration note)
        info_text = f"Vertices: {len(vertices)}\nTriangles: {len(triangles)}\nZ scale: 50x"
        plotter.add_text(info_text, font_size=10, position='lower_right')
        
        if self.verbose:
            print("=" * 50)
            print("PyVista visualization window opened")
            print("Controls:")
            print("  - Left mouse drag: Rotate")
            print("  - Mouse wheel: Zoom")
            print("  - Right mouse drag: Pan")
            print("  - Press 'q' to close window")
            print("=" * 50)
        
        plotter.show()
        
        return mesh

def load_wind_nc(nc_file_path, variable_name='wind_Prediction_0'):
    
    print("=" * 50)
    print("Reading NC file...")
    print(f"File path: {nc_file_path}")
    
    # Open NC file
    dataset = nc.Dataset(nc_file_path, 'r')
    
    # Print all variable names (for debugging)
    print(f"\nAvailable variables: {list(dataset.variables.keys())}")
    
    # Read wind speed data
    wind_data = dataset.variables[variable_name][:]
    
    # Try to read coordinate information (may have different naming)
    coord_vars = list(dataset.variables.keys())
    
    # Find X coordinate
    x_var_name = None
    for name in ['x', 'X', 'lon', 'longitude', 'Longitude']:
        if name in coord_vars:
            x_var_name = name
            break
    
    # Find Y coordinate
    y_var_name = None
    for name in ['y', 'Y', 'lat', 'latitude', 'Latitude']:
        if name in coord_vars:
            y_var_name = name
            break
    
    # Find Z coordinate
    z_var_name = None
    for name in ['z', 'Z', 'level', 'height', 'altitude']:
        if name in coord_vars:
            z_var_name = name
            break
    
    print(f"Identified coordinate variables: X='{x_var_name}', Y='{y_var_name}', Z='{z_var_name}'")
    
    # Read coordinate data
    if x_var_name:
        x_coords = dataset.variables[x_var_name][:]
    else:
        print("  X coordinate not found, using indices instead")
        x_coords = np.arange(wind_data.shape[2])
    
    if y_var_name:
        y_coords = dataset.variables[y_var_name][:]
    else:
        print("  Y coordinate not found, using indices instead")
        y_coords = np.arange(wind_data.shape[1])
    
    if z_var_name:
        z_coords = dataset.variables[z_var_name][:]
    else:
        print("  Z coordinate not found, using indices instead")
        z_coords = np.arange(wind_data.shape[0])
    
    print(f"\nData shape: {wind_data.shape}")
    print(f"X dimension: {len(x_coords)} points, range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"Y dimension: {len(y_coords)} points, range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    print(f"Z dimension: {len(z_coords)} points, range: [{z_coords.min():.2f}, {z_coords.max():.2f}]")
    
    # Calculate spacing
    # Method 1: Calculate from coordinate differences
    if len(x_coords) > 1:
        dx_calc = float(np.abs(x_coords[1] - x_coords[0]))
        # Check if geographic coordinates: value between -180 and 180 and less than 10
        is_geographic_x = (x_coords.min() >= -180 and x_coords.max() <= 180 and dx_calc < 10)
        
        if is_geographic_x:
            # Convert longitude to meters: 1 degree is about 111km * cos(latitude)
            # Calculate average latitude
            if len(y_coords) > 0:
                avg_lat = float((y_coords.min() + y_coords.max()) / 2)
            else:
                avg_lat = 40.0  # Beijing latitude
            
            dx_calc = dx_calc * 111000 * np.cos(np.radians(avg_lat))
            print(f"  Detected longitude coordinate (X range: {x_coords.min():.2f} deg to {x_coords.max():.2f} deg)")
            print(f"  Converted to meters: {dx_calc:.2f} m (based on latitude {avg_lat:.2f} deg)")
    else:
        dx_calc = 250.0  # Default value
    
    if len(y_coords) > 1:
        dy_calc = float(np.abs(y_coords[1] - y_coords[0]))
        # Check if latitude coordinate
        is_geographic_y = (y_coords.min() >= -90 and y_coords.max() <= 90 and dy_calc < 10)
        
        if is_geographic_y:
            dy_calc = dy_calc * 111000  # 1 degree latitude is about 111km
            print(f"  Detected latitude coordinate (Y range: {y_coords.min():.2f} deg to {y_coords.max():.2f} deg)")
            print(f"  Converted to meters: {dy_calc:.2f} m")
    else:
        dy_calc = 250.0
    
    if len(z_coords) > 1:
        dz_calc = float(np.abs(z_coords[1] - z_coords[0]))
    else:
        dz_calc = 10.0
    
    # Method 2: Infer from filename (your filename contains 250m)
    dx = dx_calc if dx_calc > 0 else 250.0
    dy = dy_calc if dy_calc > 0 else 250.0
    dz = dz_calc if dz_calc > 0 else 10.0
    
    # If calculated spacing is too small or abnormal, use default values
    if dx < 1.0:  # Less than 1 meter is considered abnormal
        print(f"  X direction spacing abnormal ({dx:.6f} m), using default 250m")
        dx = 250.0
    if dy < 1.0:  # Less than 1 meter is considered abnormal
        print(f"  Y direction spacing abnormal ({dy:.6f} m), using default 250m")
        dy = 250.0
    if dz < 0.1:  # Less than 0.1 meter is considered abnormal
        print(f"  Z direction spacing abnormal ({dz:.6f} m), using default 10m")
        dz = 10.0
    
    print(f"\n Final spacing: dx={dx:.2f} m, dy={dy:.2f} m, dz={dz:.2f} m")
    
    # Handle missing values
    if hasattr(dataset.variables[variable_name], '_FillValue'):
        fill_value = dataset.variables[variable_name]._FillValue
        # Replace missing values with nan, not 0
        wind_data = np.ma.filled(wind_data, fill_value=np.nan)
    else:
        fill_value = None
    
    # Convert to numpy array
    field = np.array(wind_data, dtype=np.float32)
    
    # Calculate statistics for valid data (ignore nan)
    valid_data = field[~np.isnan(field)]
    if len(valid_data) > 0:
        print(f"Wind speed statistics: min={np.nanmin(field):.2f}, max={np.nanmax(field):.2f}, mean={np.nanmean(field):.2f}")
        print(f"Valid data points: {len(valid_data)} / {field.size} ({len(valid_data)/field.size*100:.1f}%)")
    else:
        print(f"  Warning: No valid data!")
    
    # Replace nan with 0 (original behavior, avoids nan propagation).
    invalid_mask = np.isnan(field)
    field = np.nan_to_num(field, nan=0.0)
    
    # Metadata
    metadata = {
        'variable_name': variable_name,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'z_coords': z_coords,
        'fill_value': fill_value,
        'invalid_mask': invalid_mask,
        'x_var_name': x_var_name,
        'y_var_name': y_var_name,
        'z_var_name': z_var_name
    }
    
    dataset.close()
    print("=" * 50)
    
    return field, (dx, dy, dz), metadata


# Usage example
if __name__ == "__main__":
    # ========== Method 1: Use real Beijing wind speed data ==========
    print("\n" + "=" * 50)
    print("GSDMC Wind Isosurface Extraction")
    print("=" * 50 + "\n")
    
    # NC file path
    nc_file = "./sample_data.nc"
    
    try:
        # 1. Load NC file
        field, spacing, metadata = load_wind_nc(nc_file)
        
        # 2. Initialize GSDMC
        print("\nInitializing GSDMC algorithm...")
        gsdmc = GSDMC(field, spacing=spacing, sigma=0.4, verbose=True)
        
        # 3. Extract isosurface (e.g., extract isosurface for wind speed 8.0 m/s)
        isovalue = 8.0
        print(f"\nExtracting isosurface: wind speed = {isovalue} m/s")
        vertices, triangles = gsdmc.extract_isosurface(isovalue=isovalue)
        
        # 4. Export OBJ file (optional)
        output_file = f"beijing_wind_{isovalue}ms.obj"
        gsdmc.export_obj(vertices, triangles, output_file)
        
        # 5. PyVista visualization
        print("\nStarting PyVista visualization...")
        gsdmc.visualize_with_pyvista(
            vertices, 
            triangles, 
            title=f"Beijing Wind Isosurface ({isovalue} m/s)",
            show_edges=False,
            color='lightblue',
            opacity=0.8
        )
        
    except FileNotFoundError:
        print(f"\n  File not found: {nc_file}")
        print("Will run test data example...\n")
        
        # ========== Method 2: Test data (sphere) ==========
        print("Generating test data (sphere scalar field)...")
        nx, ny, nz = 50, 50, 50
        x = np.linspace(-5, 5, nx)
        y = np.linspace(-5, 5, ny)
        z = np.linspace(-5, 5, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        field = np.sqrt(X**2 + Y**2 + Z**2)  # Sphere
        
        # Run GSDMC
        gsdmc = GSDMC(field, spacing=(1.0, 1.0, 1.0), sigma=0.4)
        vertices, triangles = gsdmc.extract_isosurface(isovalue=3.0)
        
        # Export and visualize
        gsdmc.export_obj(vertices, triangles, "test_sphere.obj")
        gsdmc.visualize_with_pyvista(
            vertices, 
            triangles, 
            title="Test Sphere (isovalue=3.0)",
            show_edges=True,
            color='cyan'
        )
        
        print("\n Test complete!")


def _process_chunk_worker(indices_chunk, field, G, S, grads, spacing, T_G, T_S, isovalue):
    vertices = []
    triangles = []
    vertex_cache = {}

    nz, ny, nx = field.shape
    spacing_with_exaggeration = np.array(
        [spacing[0], spacing[1], spacing[2] * 50],
        dtype=np.float32
    )

    def node_id(i, j, k):
        return k * (ny * nx) + j * nx + i

    def edge_key(i1, j1, k1, i2, j2, k2):
        node_a = node_id(i1, j1, k1)
        node_b = node_id(i2, j2, k2)
        return (min(node_a, node_b), max(node_a, node_b))

    for k, j, i in indices_chunk:
        cube_values = np.zeros(8, dtype=np.float32)
        for vi, (di, dj, dk) in enumerate(CUBE_VERTICES):
            ii, jj, kk = i + di, j + dj, k + dk
            if ii < nx and jj < ny and kk < nz:
                cube_values[vi] = field[kk, jj, ii]

        cube_idx = 0
        for vi in range(8):
            if cube_values[vi] < isovalue:
                cube_idx |= (1 << vi)

        if cube_idx == 0 or cube_idx == 255:
            continue

        edge_flags = EDGE_TABLE[cube_idx]
        if edge_flags == 0:
            continue

        vert_indices = [-1] * 12
        for edge_id in range(12):
            if not (edge_flags & (1 << edge_id)):
                continue

            v1_local, v2_local = EDGE_VERTICES[edge_id]
            di1, dj1, dk1 = CUBE_VERTICES[v1_local]
            di2, dj2, dk2 = CUBE_VERTICES[v2_local]

            i1, j1, k1 = i + di1, j + dj1, k + dk1
            i2, j2, k2 = i + di2, j + dj2, k + dk2
            cache_key = edge_key(i1, j1, k1, i2, j2, k2)

            if cache_key in vertex_cache:
                vert_indices[edge_id] = vertex_cache[cache_key]
                continue

            p1 = np.array([i1, j1, k1], dtype=np.float32) * spacing_with_exaggeration
            p2 = np.array([i2, j2, k2], dtype=np.float32) * spacing_with_exaggeration
            v1 = field[k1, j1, i1]
            v2 = field[k2, j2, i2]

            g_avg = (G[k1, j1, i1] + G[k2, j2, i2]) / 2.0
            s_avg = (S[k1, j1, i1] + S[k2, j2, i2]) / 2.0

            if (g_avg <= T_G) and (s_avg >= T_S):
                vertex_pos = midpoint_interpolation(p1, p2)
            else:
                grad1 = np.array([
                    grads[0][k1, j1, i1],
                    grads[1][k1, j1, i1],
                    grads[2][k1, j1, i1]
                ], dtype=np.float32)
                grad2 = np.array([
                    grads[0][k2, j2, i2],
                    grads[1][k2, j2, i2],
                    grads[2][k2, j2, i2]
                ], dtype=np.float32)
                vertex_pos = hermite_interpolation(p1, p2, v1, v2, grad1, grad2, isovalue)

            new_id = len(vertices)
            vertices.append(vertex_pos)
            vertex_cache[cache_key] = new_id
            vert_indices[edge_id] = new_id

        tri_config = TRIANGLE_TABLE[cube_idx]
        for tri_idx in range(0, len(tri_config), 3):
            if tri_idx + 2 < len(tri_config):
                triangles.append([
                    vert_indices[tri_config[tri_idx]],
                    vert_indices[tri_config[tri_idx + 1]],
                    vert_indices[tri_config[tri_idx + 2]]
                ])

    if vertices:
        vertices = np.array(vertices, dtype=np.float32)
        triangles = np.array(triangles, dtype=np.int32)
    else:
        vertices = np.array([], dtype=np.float32).reshape(0, 3)
        triangles = np.array([], dtype=np.int32).reshape(0, 3)

    return vertices, triangles

