import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1. Define the 3D grid and phantom volume
##############################################################################
# Grid size
def ACR_phantom(square = 256, NZ=200, z_position = 20):
    square = square
    NX = square + 1  # number of pixels in x
    NY = square + 1  # number of pixels in y
    NZ = NZ  # number of slices in z (each slice = 1 mm thickness -> 160 mm total)

    # Physical size (mm)
    phantom_diameter_mm = 200.0  # 20 cm
    phantom_z_size_mm = 160.0
    voxel_size_x = phantom_diameter_mm / NX  # mm per pixel in x
    voxel_size_y = phantom_diameter_mm / NY  # mm per pixel in y
    voxel_size_z = phantom_z_size_mm / NZ  # mm per slice in z

    # Construct coordinate arrays
    # We'll let x,y range from -100 mm to +100 mm
    x_vals = np.linspace(-103, 103, NX, endpoint=False) + 0.5 * voxel_size_x
    y_vals = np.linspace(-103, 103, NY, endpoint=False) + 0.5 * voxel_size_y
    z_vals = np.linspace(0, 160, NZ, endpoint=False) + 0.5 * voxel_size_z

    # Create a mesh for x,y only (we will handle z separately)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

    # Initialize the phantom volume with air (-1000 HU)
    phantom_3d = np.full((NZ, NX, NY), -1000.0, dtype=np.float32)

    ##############################################################################
    # 2. Define helper functions
    ##############################################################################
    def in_cylinder(x, y, z, center_x, center_y, z_range, radius):
        """
        Returns a boolean mask indicating which (x,y,z) points lie inside a
        vertical cylinder of given radius and z-extent.
        - (center_x, center_y) : cylinder center in x-y plane
        - z_range = (z_min, z_max)
        - radius : cylinder radius in mm
        """
        inside_xy = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        inside_z = (z >= z_range[0]) & (z < z_range[1])
        return inside_xy & inside_z

    def in_sphere(x, y, z, center_x, center_y, center_z, radius):
        """
        Returns True if (x,y,z) is within the specified sphere.
        """
        dist_sq = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        return dist_sq <= radius**2

    def circle_mask_2d(x, y, radius):
        """
        Returns a 2D boolean mask for points within a circle of 'radius'
        centered at (0,0).
        """
        return (x**2 + y**2) <= radius**2

    ##############################################################################
    # 3. Define module z-boundaries (each module is 40 mm thick)
    ##############################################################################
    module1_z = (0.0, 40.0)
    module2_z = (40.0, 80.0)
    module3_z = (80.0, 120.0)
    module4_z = (120.0, 160.0)
    module4_z_bar = [(120.0, 158.0), (158.0, 160.0)]

    ##############################################################################
    # 4. Fill each module with a "background" inside the 20 cm circular phantom
    ##############################################################################
    # We'll define a 2D circular mask for the cross-section (radius = 100 mm).
    phantom_radius = 100.0
    circular_mask_2d = circle_mask_2d(X, Y, phantom_radius)

    def fill_module_background(z_start, z_end, background_hu):
        """
        Fill the specified z-range (z_start to z_end) with 'background_hu' inside
        the 20 cm diameter circle. Outside remains air (-1000).
        """
        # Convert z_start, z_end to slice indices
        slice_start = int(z_start / voxel_size_z)
        slice_end   = int(z_end   / voxel_size_z)
        # Fill
        phantom_3d[slice_start:slice_end,circular_mask_2d] = background_hu

    # Module 1 background: water-equivalent (0 HU)
    fill_module_background(*module1_z, background_hu=0.0)

    # Module 2 background: ~90 HU for low contrast
    fill_module_background(*module2_z, background_hu=90.0)

    # Module 3 background: uniform, tissue-equivalent (~0 HU)
    fill_module_background(*module3_z, background_hu=0.0)

    # Module 4 background: water-equivalent (~0 HU) for high-contrast patterns
    fill_module_background(*module4_z, background_hu=0.0)

    ##############################################################################
    # 5. Module 1 Features:
    #    - Four 1 mm steel BBs at the perimeter (3,6,9,12 o'clock)
    #    - Cylinders of Bone (~+955 HU), Polyethylene (~-95 HU),
    #      Acrylic (~+120 HU), Air (~-1000 HU), Water (~0 HU)
    #    - Each cylinder diameter = 25 mm except water cylinder = 50 mm
    #    - All extend through z = [0, 40] mm
    ##############################################################################
    # (a) Steel BBs (1 mm diameter => radius=0.5 mm) at ~the outer edge
    bb_radius = 0.5
    bb_positions = [
        ( 99.5,  0.0),  # near +x
        (-99.5,  0.0),  # near -x
        ( 0.0,  99.5),  # near +y
        ( 0.0, -99.5)   # near -y
    ]

    z_mid_module1 = 20.0  # midpoint of Module 1 in z
    for (cx, cy) in bb_positions:
        # We will set these BBs to some high HU to represent steel, e.g. 3000 HU
        # For a real phantom, these are typically visible metal beads.
        for iz, zc in enumerate(z_vals):
            if (zc >= module1_z[0]) and (zc < module1_z[1]):
                mask = in_sphere(X, Y, zc, cx, cy, z_mid_module1, bb_radius)
                phantom_3d[iz][mask] = 3000.0  # steel

    # (b) CT number cylinders:
    #     Place them in a cross-like pattern or any arrangement you prefer.
    #     For demonstration, let's place them around the center, spaced by 35 mm.
    materials = {
        "bone":        955.0,
        "poly":       -95.0,
        "acrylic":    120.0,
        "air":      -1000.0,
        "water_big":    0.0
    }
    # Center positions for the 25 mm cylinders (radius=12.5) except water_big=25 mm radius
    cylinder_radius_small = 12.5
    cylinder_radius_large = 25.0  # for water

    # Example arrangement (top, left, right, bottom, center)
    cyl_positions = {
        "bone":     (  35.0, -35.0),
        "poly":     (  -35.0,-35.0),
        "acrylic":  (  -35.0, 35.0),
        "air":      (  35.0, 35.0),
        "water_big":(  0.0,   0.0)
    }

    for mat_name, hu_val in materials.items():
        (cx, cy) = cyl_positions[mat_name]
        # Determine radius
        if mat_name == "water_big":
            rad = cylinder_radius_large
        else:
            rad = cylinder_radius_small
        
        for iz, zc in enumerate(z_vals):
            if module1_z[0] <= zc < module1_z[1]:
                mask = in_cylinder(X, Y, zc, cx, cy, module1_z, rad)
                phantom_3d[iz][mask] = hu_val

    ##############################################################################
    # 6. Module 2 Features (Low-contrast resolution):
    #    - Background ~90 HU
    #    - Series of cylinders (2, 3, 4, 5, 6 mm) at +6 HU above background => 96 HU
    #    - 4 cylinders per diameter, spaced by the cylinder diameter
    #    - A 25 mm cylinder also at +6 HU
    ##############################################################################
    low_contrast_delta = 6.0
    cyl_diameters = [2, 3, 4, 5, 6]  # mm
    z_range_mod2 = module2_z

    # We'll arrange them in a simple arc around the center
    arc_radius = 40.0  # mm from center
    arc_center = (0.0, 0.0)
    start_angle = -45  # degrees
    angle_step = 15    # spacing in degrees

    current_angle = start_angle
    for d_mm in cyl_diameters:
        for i in range(4):  # 4 cylinders of each diameter
            # Compute center
            cx = arc_radius * np.cos(np.deg2rad(current_angle))
            cy = arc_radius * np.sin(np.deg2rad(current_angle))
            r  = d_mm / 2.0
            
            for iz, zc in enumerate(z_vals):
                if z_range_mod2[0] <= zc < z_range_mod2[1]:
                    mask = in_cylinder(X, Y, zc, cx, cy, z_range_mod2, r)
                    phantom_3d[iz][mask] = 90.0 + low_contrast_delta  # ~96 HU

            current_angle += angle_step

    # Also place one 25 mm cylinder for verification
    large_contrast_radius = 12.5
    cx_large, cy_large = (0.0, -40.0)  # an example location near top
    for iz, zc in enumerate(z_vals):
        if z_range_mod2[0] <= zc < z_range_mod2[1]:
            mask = in_cylinder(X, Y, zc, cx_large, cy_large, z_range_mod2, large_contrast_radius)
            phantom_3d[iz][mask] = 90.0 + low_contrast_delta  # ~96 HU

    ##############################################################################
    # 7. Module 3 Features (Uniformity):
    #    - Background ~0 HU
    #    - Two small BBs (0.28 mm each) for distance measurement
    ##############################################################################
    bb_radius_small = 0.14  # mm
    bb_positions_mod3 = [
        ( 30.0,  -40.0),
        (-40.0, 30.0)
    ] # make the distance to 100mm
    z_mid_module3 = 100.0  # midpoint of module 3 in z

    for (cx, cy) in bb_positions_mod3:
        for iz, zc in enumerate(z_vals):
            if module3_z[0] <= zc < module3_z[1]:
                mask = in_sphere(X, Y, zc, cx, cy, z_mid_module3, bb_radius_small)
                # print(mask.max())
                phantom_3d[iz][mask] = 3000.0  # steel or metal marker

    ##############################################################################
    # 8. Module 4 Features (High-contrast resolution):
    #    - 8 bar patterns (4,5,6,7,8,9,10,12 lp/cm)
    #    - Typically small aluminum bars in 15x15 mm squares.
    #    - We'll place placeholders: squares of +1000 HU or so, just as a demo.
    ##############################################################################
    z_range_mod4 = module4_z_bar[0]
    bar_patterns = [4, 5, 6, 7, 8, 9, 10, 12]  # lp/cm
    square_size = 15.0  # mm
    cx_base, cy_base = (0.0, 0.0)
    spacing = 25.0  # mm spacing between squares

    def in_square(x, y, center_x, center_y, half_size):
        """Check if (x,y) is inside a square of side=2*half_size, centered at (cx,cy)."""
        return (abs(x - center_x) <= half_size) & (abs(y - center_y) <= half_size)

    def in_diamond(x, y, center_x, center_y, half_size):
        """Check if (x, y) is inside a diamond (rhombus) centered at (center_x, center_y) with a given half-size."""
        return abs(x - center_x) + abs(y - center_y) <= half_size

    # Arrange squares around a circle
    radius_bar_arc = 40.0
    angle_bar_step = 360.0 / len(bar_patterns)
    angle_bar_start = 0.0

    for i, lp_val in enumerate(bar_patterns):
        angle_deg = angle_bar_start + i*angle_bar_step
        cx = radius_bar_arc * np.cos(np.deg2rad(angle_deg))
        cy = radius_bar_arc * np.sin(np.deg2rad(angle_deg))
        
        half_s = square_size / 2.0
        
        for iz, zc in enumerate(z_vals):
            if z_range_mod4[0] <= zc < z_range_mod4[1]:
                # We'll mark the entire square region as a high HU placeholder (e.g. +1000)
                inside_sq = in_diamond(X, Y, cx, cy, half_s)
                phantom_3d[iz][inside_sq] = 1000.0

    # Also place 4 steel beads (1 mm) in module 4 at the perimeter
    z_mid_module4 = 140.0
    for (cx, cy) in bb_positions:  # reuse the same perimeter positions as module 1
        for iz, zc in enumerate(z_vals):
            if module4_z[0] <= zc < module4_z[1]:
                mask = in_sphere(X, Y, zc, cx, cy, z_mid_module4, bb_radius)
                phantom_3d[iz][mask] = 3000.0

    iz = int((z_position - z_vals[0]) / voxel_size_z)
    slice_img = phantom_3d[iz,:,:]

    return slice_img




import numpy as np
import matplotlib.pyplot as plt

def iodine_phantom(square = 256, NZ = 200, z_position = 20):
    ##############################################################################
    # 1. Define the 3D grid and phantom volume
    ##############################################################################
    # Grid size
    z_position = z_position
    square = square
    NX = square + 1  # number of pixels in x
    NY = square + 1  # number of pixels in y
    NZ = NZ  # number of slices in z (each slice = 1 mm thickness -> 160 mm total)

    # Physical size (mm)
    phantom_diameter_mm = 200.0  # 20 cm
    phantom_z_size_mm = 160.0
    voxel_size_x = phantom_diameter_mm / NX  # mm per pixel in x
    voxel_size_y = phantom_diameter_mm / NY  # mm per pixel in y
    voxel_size_z = phantom_z_size_mm / NZ  # mm per slice in z

    # Construct coordinate arrays
    x_vals = np.linspace(-103, 103, NX, endpoint=False) + 0.5 * voxel_size_x
    y_vals = np.linspace(-103, 103, NY, endpoint=False) + 0.5 * voxel_size_y
    z_vals = np.linspace(0, 160, NZ, endpoint=False) + 0.5 * voxel_size_z

    # Create a mesh for x,y only
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

    # Initialize the phantom volume with air (-1000 HU)
    phantom_3d = np.full((NZ, NX, NY), -1000.0, dtype=np.float32)

    ##############################################################################
    # 2. Define helper functions
    ##############################################################################
    def in_cylinder(x, y, z, center_x, center_y, z_range, radius):
        inside_xy = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        inside_z = (z >= z_range[0]) & (z < z_range[1])
        return inside_xy & inside_z

    def in_sphere(x, y, z, center_x, center_y, center_z, radius):
        dist_sq = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        return dist_sq <= radius**2

    def circle_mask_2d(x, y, radius):
        return (x**2 + y**2) <= radius**2

    ##############################################################################
    # 3. Define module z-boundary (Module 1: [0, 40] mm)
    ##############################################################################
    module1_z = (0.0, 40.0)

    ##############################################################################
    # 4. Fill Module 1 with background inside the 20 cm circular phantom
    ##############################################################################
    phantom_radius = 100.0
    circular_mask_2d = circle_mask_2d(X, Y, phantom_radius)

    def fill_module_background(z_start, z_end, background_hu):
        slice_start = int(z_start / voxel_size_z)
        slice_end   = int(z_end   / voxel_size_z)
        phantom_3d[slice_start:slice_end, circular_mask_2d] = background_hu

    fill_module_background(*module1_z, background_hu=0.0)

    ##############################################################################
    # 5. Module 1 Features (Steel BBs and CT Number Cylinders)
    ##############################################################################
    # (a) Steel BBs (1 mm diameter)
    bb_radius = 0.5
    bb_positions = [
        ( 99.5,  0.0),  # near +x
        (-99.5,  0.0),  # near -x
        ( 0.0,  99.5),  # near +y
        ( 0.0, -99.5)   # near -y
    ]

    z_mid_module1 = 20.0  # midpoint of Module 1 in z
    # for (cx, cy) in bb_positions:
    #     for iz, zc in enumerate(z_vals):
    #         if (zc >= module1_z[0]) and (zc < module1_z[1]):
    #             mask = in_sphere(X, Y, zc, cx, cy, z_mid_module1, bb_radius)
    #             phantom_3d[iz][mask] = 3000.0  # steel

    # (b) Water + Iodine mixture with time-dependent alpha weights
    materials = {
        "water_iodine": 500.0,
    }

    cyl_positions = {
        "water_iodine": [
            (  0.0,  50.0),  # 12 o'clock
            ( 35.0,  35.0),  # 1:30
            ( 50.0,   0.0),  # 3 o'clock
            ( 35.0, -35.0),  # 4:30
            (  0.0, -50.0),  # 6 o'clock
            (-35.0, -35.0),  # 7:30
            (-50.0,   0.0),  # 9 o'clock
            (-35.0,  35.0),  # 10:30
        ]
    }

    alpha_values = np.linspace(10, 500, len(cyl_positions["water_iodine"]))  # Different alphas for each region
    print(alpha_values)
    cylinder_radius = 12.5

    for (idx, (cx, cy)) in enumerate(cyl_positions["water_iodine"]):
        hu_val = 0.0 + alpha_values[idx]   # Water + Iodine concentration
        for iz, zc in enumerate(z_vals):
            if module1_z[0] <= zc < module1_z[1]:
                mask = in_cylinder(X, Y, zc, cx, cy, module1_z, cylinder_radius)
                phantom_3d[iz][mask] = hu_val

    iz = int((z_position - z_vals[0]) / voxel_size_z)
    t1 = phantom_3d[iz,:,:]

    phantom_radius = 100.0
    circular_mask_2d = circle_mask_2d(X, Y, phantom_radius)
    phantom_3d = np.full((NZ, NX, NY), -1000.0, dtype=np.float32)
    ##############################################################################
    # 4. Fill entire phantom with water (0 HU)
    ##############################################################################
    phantom_3d[:, circular_mask_2d] = 0.0

    ##############################################################################
    # 5. Visualization
    ##############################################################################

    bb_radius = 0.5
    bb_positions = [
        ( 99.5,  0.0),  # near +x
        (-99.5,  0.0),  # near -x
        ( 0.0,  99.5),  # near +y
        ( 0.0, -99.5)   # near -y
    ]

    z_mid_module1 = 20.0  # midpoint of Module 1 in z
    # for (cx, cy) in bb_positions:
    #     for iz, zc in enumerate(z_vals):
    #         if (zc >= module1_z[0]) and (zc < module1_z[1]):
    #             mask = in_sphere(X, Y, zc, cx, cy, z_mid_module1, bb_radius)
    #             phantom_3d[iz][mask] = 3000.0  # steel

    iz = int((z_position - z_vals[0]) / voxel_size_z)
    t0 = phantom_3d[iz,:,:]

    return t0, t1