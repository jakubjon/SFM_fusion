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
    'model': 'SIMPLE_PINHOLE',
    'params': [2000, 2016, 1512]  # focal_length, cx, cy
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