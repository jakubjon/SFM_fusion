# Painting Reconstruction Pipeline - Workflow Documentation

## Overview

The painting reconstruction pipeline is a modular system where each step can run independently, loading inputs and storing outputs. The main orchestration script (`main.py`) coordinates the execution based on the configuration in `config.py`.

## Pipeline Steps

### Step 1: Local SfM (`step1_local_sfm.py`)
- **Purpose**: Run local SfM for initial camera positions and using initial camera calibration
- **Inputs**: Photos directory with painting subdirectories
- **Outputs**: Local reconstructions and camera calibrations for each painting
- **Dependencies**: None

### Step 2: Global Calibration (`step2_global_calibration.py`)
- **Purpose**: Global camera calibration adjustment using selected distortion model radial by default
- **Inputs**: Local camera calibrations from Step 1
- **Outputs**: Global camera parameters
- **Dependencies**: Step 1

### Step 3: Recalculate Positions (`step3_recalculate_positions.py`)
- **Purpose**: Recalculate camera positions of each individual painting batch with global calibration
- **Inputs**: Local reconstructions from Step 1, global calibration from Step 2
- **Outputs**: Global reconstructions with updated camera poses
- **Dependencies**: Steps 1, 2

### Step 4: Point Cloud Generation (`step4_point_cloud_generation.py`)
- **Purpose**: Point cloud generation for each individual painting batch with global calibration
- **Inputs**: Global reconstructions from Step 3
- **Outputs**: Point cloud data and plane information for each painting
- **Dependencies**: Step 3

### Step 5: Rectification (`step5_rectification.py`)
- **Purpose**: Image low resolution rectification of all individual pictures of each individual painting batches with global calibration and creating overviews for each batch
- **Inputs**: Global reconstructions from Step 3, point cloud data from Step 4
- **Outputs**: Rectified images and overviews for each painting, size is determined as rectangular envelope of all image projections in to painting plane 
- **Dependencies**: Steps 3, 4

### Step 6: Manual ROI Selection (`step6_manual_roi_selection.py`)
- **Purpose**: Allowing user to manually select ROI for each overview
- **Inputs**: Overview images from Step 5
- **Outputs**: ROI selections for each painting
- **Dependencies**: Step 5

### Step 7: High Resolution Rectification (`step7_high_res_rectification.py`)
- **Purpose**: Generate high resolution orthorectified images individual pictures for ROI
- **Inputs**: Global reconstructions from Step 3, point cloud data from Step 4, ROI selections from Step 6
- **Outputs**: High resolution rectified images for each painting, size is determined as rectangular envelope of all roi points 
- **Dependencies**: Steps 3, 4, 6

## True Orthorectification Process - Unified Approach (Steps 5 & 7)

The true orthorectification process uses the **same core function** (`rectify_image_true_ortho_global`) in both Step 5 and Step 7, with the only difference being the **resolution and area of interest**. This unified approach ensures consistency and maintainability.

### Prerequisites (Available at Step 5)
1) **Painting Plane Parameters**: Normal vector and center point defining the painting plane in 3D space
2) **Camera Poses**: Position and orientation of all cameras for each painting
3) **Global Camera Calibration**: Optimized camera parameters (focal length, principal point, distortion coefficients)

### Core True Orthorectification Process

#### 1. **2D Coordinate System Establishment**
- Create orthonormal basis vectors (v1, v2) on the painting plane
- Establish 2D coordinate system (u, v) for mapping between 3D world and 2D plane coordinates

#### 2. **Image Normalization (Distortion Removal)**
- Apply global camera calibration to remove lens distortion
- Use SIMPLE_RADIAL model with iterative undistortion for accurate projection

#### 3. **Corner Projection to Painting Plane**
- Project all 4 corners of each original image onto the painting plane
- Convert from image coordinates to 2D plane coordinates (u, v)
- Handle cases where corners may be behind camera or outside image bounds

#### 4. **Rectangular Envelope Calculation**
- Calculate bounding box from all projected image corners
- Create rectangular grid covering the envelope area
- Adjust grid resolution based on aspect ratio for optimal sampling

#### 5. **True Orthorectification Execution**
- For each grid point on the painting plane:
  - Transform 3D point from world to camera coordinates
  - Project to image coordinates using calibrated camera model
  - Sample image with bilinear interpolation
  - Store color value in rectified image grid



