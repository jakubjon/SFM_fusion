# Painting Reconstruction Pipeline - Workflow Documentation

## Overview

The painting reconstruction pipeline has been refactored into a modular system where each step can run independently, loading inputs and storing outputs. The main orchestration script (`main.py`) coordinates the execution based on the configuration in `config.py`.

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
- **Outputs**: Rectified images and overviews for each painting
- **Dependencies**: Steps 3, 4

### Step 6: Manual ROI Selection (`step6_manual_roi_selection.py`)
- **Purpose**: Allowing user to manually select ROI for each overview
- **Inputs**: Overview images from Step 5
- **Outputs**: ROI selections for each painting
- **Dependencies**: Step 5

### Step 7: High Resolution Rectification (`step7_high_res_rectification.py`)
- **Purpose**: Generate high resolution orthorectified images individual pictures for ROI
- **Inputs**: Global reconstructions from Step 3, point cloud data from Step 4, ROI selections from Step 6
- **Outputs**: High resolution rectified images for each painting
- **Dependencies**: Steps 3, 4, 6

true ortorectification wrapup
1) we are at the beggining of step 5 so wee know painting plane parameters, position and orientation of all cameras, global camera calibration
2) 2D coordinate system of the plane is established
3) picture normalization (removing distosion) sing global camera calibration 
3) corners of the original pictures are reprojected in to the painting plane and represented as 4 2D points
4) rectangular envelop is calculated in the paintong plane and only for this area the original picture is true ortorectified using position of the camera and position of the plane 
5) overievw is created by fusing all images taken in acout relative shifts
6) step5 manual ROI selection - which basicaly define otline of the painting in painting plane 
7) step7 only for this area in panitng plane true ortorctified high resolution image is rendered

## Usage

### Run Complete Pipeline
```bash
python main.py
```

### Run Single Step
```bash
python main.py --step run_local_sfm
python main.py --step global_calibration
python main.py --step recalculate_positions
python main.py --step point_cloud_generation
python main.py --step rectification
python main.py --step manual_roi_selection
python main.py --step high_res_rectification
```

### List Available Steps
```bash
python main.py --list-steps
```

### Check Step Dependencies
```bash
python main.py --check-deps rectification
```

## Configuration

Enable/disable steps in `config.py`:

```python
PROCESSING_STEPS = {
    'run_local_sfm': True,           # Step 1
    'global_calibration': True,      # Step 2
    'recalculate_positions': True,   # Step 3
    'point_cloud_generation': True,  # Step 4
    'rectification': True,           # Step 5
    'manual_roi_selection': True,    # Step 6
    'high_res_rectification': True   # Step 7
}
```

## Output Structure

```
outputs/
├── intermediate/           # Intermediate results from each step
│   ├── step1_local_sfm_results.json
│   ├── step2_global_calibration_results.json
│   ├── step3_recalculate_positions_results.json
│   ├── step4_point_cloud_results.json
│   ├── step5_rectification_results.json
│   ├── step6_roi_selection_results.json
│   └── step7_high_res_rectification_results.json
├── rectified/             # Low resolution rectified images
│   ├── painting1_overview.jpg
│   ├── painting1_rectified_*.jpg
│   └── ...
├── high_res_rectified/    # High resolution rectified images
│   ├── painting1_high_res_*.jpg
│   └── ...
├── roi_selections/        # ROI visualization images
│   ├── painting1_roi_visualization.jpg
│   └── ...
└── point_clouds/         # Point cloud visualizations (optional)
    ├── painting1_point_cloud.png
    └── ...
```

## Step Independence

Each step can be run independently:

1. **Input Loading**: Each step loads its required inputs from the intermediate directory
2. **Output Storage**: Each step saves its results to the intermediate directory
3. **Dependency Checking**: Steps check for required inputs before execution
4. **Resume Capability**: Steps can resume from existing intermediate results

## Error Handling

- Steps check for required inputs and fail gracefully if missing
- Each step logs its progress and errors
- Failed steps don't prevent subsequent steps from running (if dependencies are met)
- Intermediate results are preserved for debugging

## Testing Individual Steps

Each step module can be tested independently:

```bash
python step1_local_sfm.py
python step2_global_calibration.py
# etc.
```

## Customization

- Modify step parameters in `config.py`
- Add new steps by creating new step modules following the `StepBase` interface
- Customize step dependencies in the main pipeline class
- Adjust output formats and file naming conventions
