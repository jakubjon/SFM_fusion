# Configuration file for Painting Reconstruction System

# Directory settings
PHOTOS_DIR = 'Photos'
OUTPUT_DIR = 'outputs'

# Processing parameters
OVERVIEW_GRID_SIZE = 512  # Resolution of rectified images (reduced for performance)
HIGH_RES_GRID_SIZE = 2048  # Resolution of rectified images (reduced for performance)
RANSAC_ITERATIONS = 100  # Number of RANSAC iterations for plane fitting
PLANE_THRESHOLD = 0.1  # Distance threshold for plane inliers (meters)

# Camera calibration
DEFAULT_CAMERA_PARAMS = {
    'width': 4032,  # Typical phone camera resolution
    'height': 3024,
    'model': 'SIMPLE_RADIAL',  # More complex model for global calibration
    'params': [2000, 2016, 1512, 0.0]  # focal_length, cx, cy, k1 (radial distortion)
}

# Processing steps configuration
PROCESSING_STEPS = {
    'run_local_sfm': False,           # Step 1: Run local SfM for initial camera positions and using initial camera calibration
    'global_calibration': False,      # Step 2: Global camera calibration adjustment (using selected distortion model radial by default)
    'recalculate_positions': False,   # Step 3: Recalculate camera positions of each individual painting batch with global calibration
    'point_cloud_generation': False,  # Step 4: Point cloud generation for each individual painting batch with global calibration
    'rectification': False,           # Step 5: True orthorectification of all individual pictures using proper camera projection with global calibration and creating overviews for each batch
    'manual_roi_selection': True,    # Step 6: Manual ROI selection with proper coordinate system conversion
    'high_res_rectification': True   # Step 7: Generate high resolution orthorectified images for ROI using true orthorectification
}