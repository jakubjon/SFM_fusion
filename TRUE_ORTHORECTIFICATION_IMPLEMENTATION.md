# True Orthorectification Implementation

## Overview

This document describes the corrected implementation of true orthorectification that addresses the critical flaws in the original codebase. The new implementation follows a rigorous approach that properly handles camera distortion, uses per-pixel sampling, and maintains consistent coordinate systems throughout the pipeline.

## Key Issues Fixed

### 1. **Proper Camera Projection with Distortion**
**Problem**: Original code used `K @ pose[:3]` which ignored radial distortion parameters.

**Solution**: Created `CameraProjector` class that uses pycolmap's proper camera model:
```python
class CameraProjector:
    def __init__(self, camera_params):
        self.camera = pycolmap.Camera(
            camera_id=1,
            model=camera_params['model'],  # SIMPLE_RADIAL
            width=camera_params['width'],
            height=camera_params['height'],
            params=camera_params['params']  # [fx, cx, cy, k1]
        )
    
    def project_point(self, point_3d):
        # Uses pycolmap's projection which handles distortion correctly
        return self.camera.project(point_3d)
```

### 2. **True Orthorectification with Per-Pixel Sampling**
**Problem**: Original code used homography-based warping which is only an approximation.

**Solution**: Implemented per-pixel sampling using proper camera projection:
```python
def rectify_image_true_ortho(image, camera_projector, pose, grid_points_3d, grid_size):
    # For each grid point on painting plane
    for i, point_3d in enumerate(grid_points_3d):
        # Transform to camera coordinates
        point_cam = R @ point_3d + t
        
        # Project to image coordinates (with distortion)
        point_2d = camera_projector.project_point(point_cam)
        
        if point_2d is not None:
            # Sample image with bilinear interpolation
            color = sample_image_bilinear(image, point_2d)
            rectified[grid_j, grid_i] = color
```

### 3. **Consistent 2D Coordinate System**
**Problem**: Coordinate system was recreated inconsistently between steps.

**Solution**: Created `PlaneProjector` class for consistent plane-to-world mapping:
```python
class PlaneProjector:
    def __init__(self, plane_normal, plane_center):
        # Create orthonormal basis for 2D coordinate system
        self._create_coordinate_system()
    
    def world_to_plane_2d(self, point_3d):
        vec = point_3d - self.plane_center
        u = np.dot(vec, self.v1)
        v = np.dot(vec, self.v2)
        return np.array([u, v])
    
    def plane_2d_to_world(self, point_2d):
        return self.plane_center + point_2d[0] * self.v1 + point_2d[1] * self.v2
```

### 4. **Proper ROI Coordinate Conversion**
**Problem**: ROI was defined in overview pixels without proper conversion to plane coordinates.

**Solution**: Implemented proper coordinate conversion in Step 6:
```python
def convert_roi_to_plane_coordinates(self, roi_points, grid_bounds, overview_size):
    # Map from overview image coordinates to grid coordinates
    grid_x = (x / overview_width) * grid_size
    grid_y = (y / overview_height) * grid_size
    
    # Map from grid coordinates to plane coordinates
    u = min_u + (grid_x / (grid_size - 1)) * (max_u - min_u)
    v = min_v + (grid_y / (grid_size - 1)) * (max_v - min_v)
    
    return [u, v]
```

### 5. **ROI-Based High-Resolution Rectification**
**Problem**: High-res rectification processed the entire painting area instead of just the ROI.

**Solution**: Created ROI-specific rectification grid:
```python
def create_roi_rectification_grid(self, plane_projector, roi_bounds_plane, target_resolution):
    # Create grid for ROI area only
    for i in range(target_resolution):
        for j in range(target_resolution):
            # Map grid coordinates to ROI plane coordinates
            u = u_min + (i / (target_resolution - 1)) * (u_max - u_min)
            v = v_min + (j / (target_resolution - 1)) * (v_max - v_min)
            
            # Convert to 3D world coordinates
            point_3d = plane_projector.plane_2d_to_world(np.array([u, v]))
```

## Implementation Steps

### Step 1-4: Foundation (Unchanged)
- Local SfM with pinhole model
- Global calibration with SIMPLE_RADIAL model
- Position recalculation with global calibration
- Point cloud generation and plane fitting

### Step 5: True Orthorectification
1. **Create Camera Projector**: Initialize with global calibration parameters
2. **Create Plane Projector**: Establish consistent 2D coordinate system
3. **Get Image Corners**: Project image corners to 3D using proper camera model
4. **Create Rectification Grid**: Generate grid on painting plane with margin
5. **Per-Pixel Sampling**: For each grid point:
   - Transform to camera coordinates
   - Project to image coordinates (with distortion)
   - Sample image with bilinear interpolation
6. **Overview Fusion**: Use weighted fusion instead of simple averaging

### Step 6: ROI Selection with Coordinate Conversion
1. **Interactive Selection**: Allow user to select ROI on overview image
2. **Coordinate Conversion**: Convert ROI from overview pixels to plane coordinates
3. **Store Both Representations**: Keep both overview and plane coordinate versions

### Step 7: High-Resolution ROI Rectification
1. **ROI Grid Creation**: Create high-resolution grid for ROI area only
2. **True Orthorectification**: Apply per-pixel sampling for ROI area
3. **Image Enhancement**: Apply CLAHE and sharpening
4. **Output**: Save 2048x2048 resolution images

## Key Benefits

### 1. **Accuracy**
- Proper handling of radial distortion
- True orthorectification instead of homography approximation
- Consistent coordinate systems throughout pipeline

### 2. **Efficiency**
- ROI-based processing reduces computation
- Reusable camera and plane projector objects
- Proper intermediate result caching

### 3. **Robustness**
- Graceful handling of projection failures
- Fallback mechanisms for missing data
- Comprehensive error checking

### 4. **Maintainability**
- Modular design with clear interfaces
- Separation of concerns between camera, plane, and rectification logic
- Well-documented coordinate transformations

## Usage

### Enable All Steps
```python
# In config.py
PROCESSING_STEPS = {
    'run_local_sfm': True,
    'global_calibration': True,
    'recalculate_positions': True,
    'point_cloud_generation': True,
    'rectification': True,
    'manual_roi_selection': True,
    'high_res_rectification': True
}
```

### Run Complete Pipeline
```bash
python main.py
```

### Run Individual Steps
```bash
python main.py --step rectification
python main.py --step manual_roi_selection
python main.py --step high_res_rectification
```

## Output Structure

```
outputs/
├── intermediate/
│   ├── step5_rectification_results.json      # True orthorectification data
│   ├── step6_roi_selection_results.json      # ROI selections with plane coordinates
│   └── step7_high_res_rectification_results.json  # High-res ROI data
├── rectified/
│   ├── painting1_overview.jpg                # Fused overview
│   └── painting1_rectified_*.jpg             # Low-res rectified images
├── high_res_rectified/
│   └── painting1_high_res_*.png              # High-res ROI images
└── roi_selections/
    └── painting1_roi_visualization.jpg       # ROI visualization
```

## Technical Details

### Camera Model
- **Model**: SIMPLE_RADIAL
- **Parameters**: [fx, cx, cy, k1]
- **Projection**: Uses pycolmap's native projection with distortion

### Coordinate Systems
- **World**: 3D COLMAP coordinate system
- **Plane**: 2D coordinate system on painting plane (u, v)
- **Image**: Pixel coordinates in original images
- **Overview**: Pixel coordinates in overview images

### Grid Resolution
- **Low-res**: 256x256 (config.GRID_SIZE // 2)
- **High-res**: 2048x2048 (target_resolution)

### Sampling Method
- **Bilinear Interpolation**: For smooth sampling from original images
- **Boundary Checking**: Ensures sampled points are within image bounds
- **Error Handling**: Graceful handling of projection failures

## Validation

The corrected implementation addresses all the major issues identified:

1. ✅ **Radial distortion properly handled** using pycolmap's camera model
2. ✅ **Per-pixel sampling** instead of homography approximation
3. ✅ **Consistent coordinate systems** throughout the pipeline
4. ✅ **Proper ROI mapping** from overview to plane coordinates
5. ✅ **Efficient ROI-based processing** for high-resolution output
6. ✅ **Robust error handling** and fallback mechanisms

This implementation provides true orthorectification that is both accurate and efficient, following the rigorous approach outlined in the requirements. 