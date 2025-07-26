# Configuration file for Painting Reconstruction System

# Directory settings
PHOTOS_DIR = 'Photos'
OUTPUT_DIR = 'outputs'

# Processing parameters
GRID_SIZE = 512  # Resolution of rectified images (reduced for performance)
RANSAC_ITERATIONS = 100  # Number of RANSAC iterations for plane fitting
PLANE_THRESHOLD = 0.1  # Distance threshold for plane inliers (meters)

# Camera calibration
DEFAULT_CAMERA_PARAMS = {
    'width': 4032,  # Typical phone camera resolution
    'height': 3024,
    'model': 'SIMPLE_RADIAL',  # More complex model for global calibration
    'params': [2000, 2016, 1512, 0.0]  # focal_length, cx, cy, k1 (radial distortion)
}

# Image processing
BILATERAL_FILTER_PARAMS = {
    'd': 15,  # Diameter of pixel neighborhood
    'sigma_color': 75,  # Filter sigma in color space
    'sigma_space': 75   # Filter sigma in coordinate space
}

# Optical flow parameters for image alignment
OPTICAL_FLOW_PARAMS = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

# Fusion parameters
import numpy as np
SHARPENING_KERNEL = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# Output settings
SAVE_INTERMEDIATE = True  # Save individual rectified images
SAVE_CALIBRATION = True   # Save camera calibration data
VERBOSE = True            # Print detailed progress information 

# Processing steps configuration
PROCESSING_STEPS = {
    'run_local_sfm': False,           # Step 1: Run local SfM for initial camera positions and using initial camera calibration
    'global_calibration': False,      # Step 2: Global camera calibration adjustment (using selected distortion model radial by default)
    'recalculate_positions': False,   # Step 3: Recalculate camera positions of each individual painting batch with global calibration
    'point_cloud_generation': False,  # Step 4: Point cloud generation for each individual painting batch with global calibration
    'rectification': False,           # Step 5: Image low resolution rectification of all individual pictures of each individual painting batches with global calibration and creating overviews for each batch
    'manual_roi_selection': False,    # Step 6: Allowing user to manually select ROI for each overview
    'high_res_rectification': True   # Step 7: Generate high resolution orthorectified images individual pictures for ROI
}

# Execution control
EXECUTION_CONTROL = {
    'overwrite_existing': True,      # Whether to overwrite existing intermediate results
    'force_recalculation': True      # Whether to force recalculation even if data exists
}

# Intermediate results configuration
INTERMEDIATE_RESULTS = {
    'save_intermediate': True,              # Enable/disable all intermediate result saving
    'save_local_reconstructions': True,    # Save local SfM reconstructions
    'save_global_calibration': True,       # Save global calibration results
    'save_point_clouds': True,             # Save point cloud data
    'save_rectification_data': True,       # Save rectification parameters
    'save_overview_data': True             # Save overview generation data
}

# Rectification configuration
RECTIFICATION_CONFIG = {
    'use_2d_coordinate_system': True,      # Use proper 2D coordinate system
    'create_rectangular_envelope': True,   # Create rectangular envelope
    'show_original_frames': True,          # Show original frame outlines
    'reduced_resolution_factor': 0.5,      # Resolution reduction factor
    'envelope_margin': 0.1                 # Margin around painting (10%)
} 