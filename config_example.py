# Example configuration file showing how to customize processing steps
# Copy this to config.py and modify as needed

# Processing steps configuration
PROCESSING_STEPS = {
    'run_local_sfm': True,           # Step 1: Run local SfM for initial camera positions
    'global_calibration': True,      # Step 2: Global bundle adjustment
    'recalculate_positions': True,   # Step 3: Recalculate with global calibration
    'point_cloud_generation': True,  # Step 4: Point cloud generation with global calibration
    'rectification': True,           # Step 5: Image rectification with global calibration
    'create_overviews': True,        # Step 6: Create painting overviews
    'comprehensive_overview': True   # Step 7: Create comprehensive overview
}

# Example: Skip comprehensive overview
# PROCESSING_STEPS['comprehensive_overview'] = False

# Example: Only run local SfM and global calibration
# PROCESSING_STEPS = {
#     'run_local_sfm': True,
#     'global_calibration': True,
#     'recalculate_positions': False,
#     'point_cloud_generation': False,
#     'rectification': False,
#     'create_overviews': False,
#     'comprehensive_overview': False
# }

# Intermediate results configuration (automatic saving)
INTERMEDIATE_RESULTS = {
    'save_intermediate': True,              # Master switch for intermediate saving
    'save_local_reconstructions': True,    # Save local SfM reconstructions
    'save_global_calibration': True,       # Save global calibration results
    'save_point_clouds': True,             # Save point cloud data
    'save_rectification_data': True,       # Save rectification parameters
    'save_overview_data': True             # Save overview generation data
}

# Example: Disable all intermediate saving
# INTERMEDIATE_RESULTS['save_intermediate'] = False

# Rectification configuration
RECTIFICATION_CONFIG = {
    'use_2d_coordinate_system': True,      # Use proper 2D coordinate system
    'create_rectangular_envelope': True,   # Create rectangular envelope
    'show_original_frames': True,          # Show original frame outlines
    'reduced_resolution_factor': 0.5,      # Resolution reduction factor
    'envelope_margin': 0.1                 # Margin around painting (10%)
}

# Example: Disable original frame outlines
# RECTIFICATION_CONFIG['show_original_frames'] = False

# Example: Increase envelope margin
# RECTIFICATION_CONFIG['envelope_margin'] = 0.2  # 20% margin 