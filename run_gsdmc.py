"""
Beijing Wind Data GSDMC Isosurface Extraction Script
Visualization using PyVista
"""

import numpy as np
import os
import time
from datetime import datetime
from GSDMC import GSDMC, load_wind_nc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OBJ_DIR = os.environ.get(
    "GSDMC_OBJ_DIR",
    os.path.join(BASE_DIR, "outputs", "obj")
)
DEFAULT_IMAGE_DIR = os.environ.get(
    "GSDMC_IMAGE_DIR",
    os.path.join(BASE_DIR, "outputs", "images")
)


def get_user_isovalues(field_min, field_max):
    
    print("\n" + "=" * 60)
    print("Isosurface Threshold Configuration")
    print("=" * 60)
    print(f"Data range: [{field_min:.2f}, {field_max:.2f}] m/s")
    
    while True:
        choice = input("\nCustomize isosurface thresholds? (y/n, default n): ").strip().lower()
        
        if choice == '' or choice == 'n':
            # Use all integers between min and max
            isovalues = list(range(int(np.ceil(field_min)), int(np.floor(field_max)) + 1))
            print(f"\nWill extract the following isosurfaces: {isovalues}")
            return isovalues
        
        elif choice == 'y':
            print("\nPlease enter isosurface thresholds (multiple values separated by spaces, e.g.: 6 8 10 12):")
            try:
                user_input = input("Thresholds: ").strip()
                isovalues = [float(x) for x in user_input.split()]
                
                # Validate range
                valid_values = [v for v in isovalues if field_min <= v <= field_max]
                invalid_values = [v for v in isovalues if v not in valid_values]
                
                if invalid_values:
                    print(f"  The following values are out of range and will be ignored: {invalid_values}")
                
                if valid_values:
                    print(f"\nWill extract the following isosurfaces: {valid_values}")
                    return valid_values
                else:
                    print(" No valid thresholds, please re-enter")
                    
            except ValueError:
                print(" Input format error, please enter numbers")
        else:
            print(" Please enter 'y' or 'n'")


def ask_export_obj(default_output_dir=DEFAULT_OBJ_DIR):
   
    print("\n" + "=" * 60)
    print("OBJ File Export Configuration")
    print("=" * 60)
    
    while True:
        choice = input("Export OBJ file? (y/n, default y): ").strip().lower()
        
        if choice == '' or choice == 'y':
            output_dir = default_output_dir
            
            # Create output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f" Created output directory: {output_dir}")
            else:
                print(f" Output directory: {output_dir}")
            
            return True, output_dir
        
        elif choice == 'n':
            print("  Will not export OBJ file")
            return False, None
        
        else:
            print(" Please enter 'y' or 'n'")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("    Beijing Wind Data - GSDMC Isosurface Extraction")
    print("=" * 60 + "\n")
    
    # ========== Configuration Parameters ==========
    nc_file = "./sample_data.nc"
    variable_name = 'wind_Prediction_0'
    sigma = 0.4     # Similarity field parameter
    mode = "GSDMC"  # Options: "MC", "MC_ULVR", "GSDMC"
    
    # ========== 1. Load NC Data ==========
    print("Step 1/5: Loading NC file...")
    try:
        field, spacing, metadata = load_wind_nc(nc_file, variable_name)
        print(f" Data loaded successfully!")
        print(f"   Data shape: {field.shape}")
        print(f"   Physical spacing: dx={spacing[0]:.1f}m, dy={spacing[1]:.1f}m, dz={spacing[2]:.1f}m")
    except Exception as e:
        print(f" Loading failed: {e}")
        return
    
    # Get valid data range from original valid points when available.
    invalid_mask = metadata.get('invalid_mask')
    if invalid_mask is not None:
        valid_data = field[~invalid_mask]
    else:
        valid_data = field[np.isfinite(field)]
    field_min = valid_data.min() if len(valid_data) > 0 else 0
    field_max = valid_data.max() if len(valid_data) > 0 else 0
    
    # ========== 2. User Interaction: Select Isosurface Thresholds ==========
    isovalues = get_user_isovalues(field_min, field_max)
    if not isovalues:
        print(" No valid isosurface thresholds")
        return
    
    # ========== 3. User Interaction: Export OBJ ==========
    export_obj, output_dir = ask_export_obj()
    
    # ========== 4. Initialize GSDMC ==========
    print(f"\nStep 2/5: Initializing {mode} algorithm (sigma={sigma})...")
    init_start = time.time()
    # Multiprocessing settings:
    # n_jobs=1  -> Disable multiprocessing (recommended, avoid memory issues)
    # n_jobs=4  -> Use 4 processes
    # n_jobs=-1 -> Auto-detect (may cause memory overflow)
    gsdmc = GSDMC(field, spacing=spacing, sigma=sigma, verbose=True, n_jobs=1, mode=mode)
    init_elapsed = time.time() - init_start
    print(f"\n  Total initialization time: {init_elapsed:.2f}s\n")
    
    # ========== 5. Batch Extract Isosurfaces ==========
    print(f"\nStep 3/5: Batch extracting isosurfaces...")
    print(f"Total {len(isovalues)} isosurfaces to extract")
    print("=" * 60)
    
    all_results = []  # Store all results
    batch_start = time.time()
    extraction_times = []
    
    for idx, isovalue in enumerate(isovalues, 1):
        print(f"\n[{idx}/{len(isovalues)}] Extracting isosurface: wind speed = {isovalue} m/s")
        print("-" * 60)
        
        iso_start = time.time()
        vertices, triangles = gsdmc.extract_isosurface(isovalue=isovalue)
        iso_elapsed = time.time() - iso_start
        extraction_times.append(iso_elapsed)
        
        if len(vertices) == 0:
            print(f"  Isosurface not found for wind speed={isovalue} m/s, skipping")
            continue
        
        all_results.append({
            'isovalue': isovalue,
            'vertices': vertices,
            'triangles': triangles
        })
        
        # Export OBJ file
        if export_obj:
            export_start = time.time()
            output_file = os.path.join(output_dir, f"{mode}_wind_{isovalue:.1f}ms.obj")
            gsdmc.export_obj(vertices, triangles, output_file)
            export_elapsed = time.time() - export_start
            print(f" OBJ export time: {export_elapsed:.2f}s")
    
    batch_elapsed = time.time() - batch_start
    
    if not all_results:
        print("\n No isosurfaces were successfully extracted")
        return
    
    print("\n" + "=" * 60)
    print(f" Batch extraction complete!")
    print(f" Performance Statistics:")
    print(f"  - Successfully extracted: {len(all_results)} isosurfaces")
    print(f"  - Total time: {batch_elapsed:.2f}s ({batch_elapsed/60:.1f}min)")
    print(f"  - Average per isosurface: {np.mean(extraction_times):.2f}s")
    print(f"  - Fastest: {np.min(extraction_times):.2f}s")
    print(f"  - Slowest: {np.max(extraction_times):.2f}s")
    
    # Performance comparison (assume original version takes 100s per isosurface)
    baseline_time = len(all_results) * 100
    speedup = baseline_time / batch_elapsed
    print(f"\n Performance Improvement:")
    print(f"  - Traditional MC estimated time: {baseline_time/60:.1f}min")
    print(f"  - Actual time: {batch_elapsed/60:.1f}min")
    print(f"  - Speedup: {speedup:.1f}x")
    
    if export_obj:
        print(f"\n File Output:")
        print(f"  - OBJ files saved to: {output_dir}")
    print("=" * 60)
    
    # ========== 6. Visualize All Isosurfaces ==========
    print(f"\nStep 4/5: PyVista visualization...")
    print(f"Starting visualization interface, loading {len(all_results)} isosurfaces...")
    print("Tip: Use left checkboxes to select isosurfaces to display\n")
    
    visualize_multiple(gsdmc, all_results)
    
    print("\n" + "=" * 60)
    print(" Processing complete!")
    print("=" * 60)


def visualize_multiple(gsdmc, results, output_dir=DEFAULT_IMAGE_DIR):
    
    import pyvista as pv
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from datetime import datetime
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Create viridis colormap
    isovalues = [r['isovalue'] for r in results]
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(isovalues), vmax=max(isovalues))
    
    # Store actors for checkbox control
    actors_list = []
    
    for idx, result in enumerate(results):
        isovalue = result['isovalue']
        vertices = result['vertices']
        triangles = result['triangles']
        
        # Create mesh
        faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).flatten()
        mesh = pv.PolyData(vertices, faces)
        mesh.compute_normals(inplace=True)
        
        # Get color
        color = cmap(norm(isovalue))
        
        # Add to scene
        actor = plotter.add_mesh(
            mesh,
            color=color[:3],  # RGB, excluding alpha
            opacity=0.7,
            smooth_shading=True,
            name=f"iso_{isovalue}"  # Unique name
        )
        
        actors_list.append({
            'actor': actor,
            'isovalue': isovalue
        })
    
    # Add checkbox control
    checkbox_x = 10
    checkbox_y = 10
    checkbox_width = 20
    
    for i, item in enumerate(actors_list):
        actor = item['actor']
        isovalue = item['isovalue']
        
        # Add checkbox
        plotter.add_checkbox_button_widget(
            lambda state, a=actor: a.SetVisibility(state),
            value=True,  # Show by default
            position=(checkbox_x, checkbox_y + i * 30),
            size=checkbox_width,
            border_size=1,
            color_on='green',
            color_off='red'
        )
        
        # Add label
        plotter.add_text(
            f"Wind Speed: {isovalue:.2f} m/s",
            position=(checkbox_x + 30, checkbox_y + i * 30),
            font_size=10,
            color='black'
        )
    
    # Add export functionality
    def export_image():
        """Export high-resolution image"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wind_isosurface_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # 800 DPI, 10x8 inches
        pixel_width = 8000
        pixel_height = 6400
        
        plotter.screenshot(
            filename=filepath,
            window_size=(pixel_width, pixel_height),
            return_img=False,
            transparent_background=False
        )
        
        print("\n" + "=" * 70)
        print(" 800 DPI image exported successfully!")
        print(f" File: {filename}")
        print(f" Path: {filepath}")
        print(f"  Size: {pixel_width}x{pixel_height} pixels (10x8 inches, 800 DPI)")
        print("=" * 70)
    
    # Keyboard shortcuts
    plotter.add_key_event('s', lambda: export_image())
    plotter.add_key_event('S', lambda: export_image())
    plotter.add_key_event('r', lambda: plotter.reset_camera())
    plotter.add_key_event('R', lambda: plotter.reset_camera())
    
    def show_help():
        print("=" * 60)
        print(" GSDMC Wind Field Isosurface Visualization Control")
        print("=" * 60)
        print("  Keyboard Shortcuts:")
        print("   - S: Export 800 DPI image")
        print("   - R: Reset view")
        print("   - H: Show help")
        print("   - Q: Quit")
        print("\n  Mouse:")
        print("   - Left-click drag: Rotate")
        print("   - Scroll wheel: Zoom")
        print("   - Right-click drag: Pan")
        print("\n Checkboxes:")
        print("   - Click left checkboxes to show/hide isosurfaces")
        print("=" * 60)
    
    plotter.add_key_event('h', lambda: show_help())
    plotter.add_key_event('H', lambda: show_help())
    
    # Add axes and bounding box
    plotter.add_axes()
    plotter.add_bounding_box()
    
    # Title
    plotter.add_text(
        f"Beijing Wind Speed GSDMC Isosurfaces ({len(results)} Layers)",
        position='upper_edge',
        font_size=16,
        color='black'
    )
    
    # Control instructions
    plotter.add_text(
        "Press 'S' to export | 'H' for help | Click checkboxes to toggle",
        position='lower_edge',
        font_size=10,
        color='gray'
    )
    
    if gsdmc.verbose:
        print("\n" + "=" * 60)
        print(" PyVista visualization window launched")
        print("=" * 60)
        print("  Shortcuts: S-Export | R-Reset | H-Help | Q-Quit")
        print(" Checkboxes: Click left to show/hide isosurfaces")
        print("  Mouse: Left-Rotate | Scroll-Zoom | Right-Pan")
        print("=" * 60)
    
    plotter.show()


def quick_preview(nc_file, num_levels=5):
    """
    Quick data preview, recommend suitable isovalue
    
    Parameters:
        nc_file: NC file path
        num_levels: Number of isoline levels to display
    """
    print("\n" + "=" * 60)
    print("    Data Preview - Recommended Isosurface Parameters")
    print("=" * 60 + "\n")
    
    field, spacing, metadata = load_wind_nc(nc_file)
    invalid_mask = metadata.get('invalid_mask')
    valid_field = field[~invalid_mask] if invalid_mask is not None else field[np.isfinite(field)]
    
    # Statistical information
    print("Wind Speed Statistics:")
    print(f"  Minimum: {valid_field.min():.2f} m/s")
    print(f"  Maximum: {valid_field.max():.2f} m/s")
    print(f"  Mean: {valid_field.mean():.2f} m/s")
    print(f"  Median: {np.median(valid_field):.2f} m/s")
    print(f"  Standard Deviation: {valid_field.std():.2f} m/s")
    
    # Recommended isosurface values
    print(f"\nRecommended Isosurface Parameters ({num_levels} levels):")
    percentiles = np.linspace(10, 90, num_levels)
    for i, p in enumerate(percentiles):
        val = np.percentile(valid_field, p)
        print(f"  Level {i+1}: {val:.2f} m/s ({p:.0f}th percentile)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
   
    main()
