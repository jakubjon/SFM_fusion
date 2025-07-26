# Painting Reconstruction System

This system reconstructs paintings from multiple photographs taken from different angles, removes perspective distortion, and creates high-quality orthophotos with reflection removal and super-resolution fusion.

## Features

- **Modular Pipeline**: Each processing step runs independently with clear inputs and outputs
- **Multi-painting support**: Process multiple painting sets simultaneously
- **Automatic camera calibration**: Estimates camera parameters from all image sets
- **Structure-from-Motion**: Uses COLMAP for robust 3D reconstruction
- **Perspective correction**: Removes camera distortion and perspective effects
- **Reflection removal**: Advanced filtering to reduce reflections and glare
- **Super-resolution fusion**: Combines multiple images for higher quality results
- **Orthophoto generation**: Creates true orthographic projections of paintings
- **ROI-based processing**: Manual selection of regions of interest for high-resolution output

## Pipeline Steps

The system is organized into 7 modular steps:

1. **Local SfM**: Run local SfM for initial camera positions and using initial camera calibration
2. **Global Calibration**: Global camera calibration adjustment using selected distortion model radial by default
3. **Recalculate Positions**: Recalculate camera positions of each individual painting batch with global calibration
4. **Point Cloud Generation**: Point cloud generation for each individual painting batch with global calibration
5. **Rectification**: Image low resolution rectification of all individual pictures of each individual painting batches with global calibration and creating overviews for each batch
6. **Manual ROI Selection**: Allowing user to manually select ROI for each overview
7. **High Resolution Rectification**: Generate high resolution orthorectified images individual pictures for ROI

## Directory Structure

```
SFM_fusion/
├── Photos/
│   ├── 1/          # Painting set 1
│   │   ├── PXL_20250622_092601307.jpg
│   │   ├── PXL_20250622_092604380.jpg
│   │   └── ...
│   ├── 2/          # Painting set 2
│   ├── 3/          # Painting set 3
│   └── 4/          # Painting set 4
├── main.py          # Main orchestration script
├── step_base.py     # Base class for all processing steps
├── step1_local_sfm.py              # Step 1: Local SfM
├── step2_global_calibration.py     # Step 2: Global calibration
├── step3_recalculate_positions.py  # Step 3: Position recalculation
├── step4_point_cloud_generation.py # Step 4: Point cloud generation
├── step5_rectification.py          # Step 5: Low resolution rectification
├── step6_manual_roi_selection.py  # Step 6: Manual ROI selection
├── step7_high_res_rectification.py # Step 7: High resolution rectification
├── config.py        # Configuration file
├── requirements.txt # Python dependencies
├── workflow.md      # Detailed workflow documentation
└── README.md        # This file
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install COLMAP (required for pycolmap):
   - **Windows**: Download from https://colmap.github.io/
   - **Linux**: `sudo apt-get install colmap`
   - **macOS**: `brew install colmap`

## Usage

### 1. Prepare your photos:
   - Create folders for each painting (1, 2, 3, 4, etc.)
   - Place photos of each painting in its respective folder
   - Use the same camera for all photos
   - Take photos from different angles (15-30° apart recommended)

### 2. Configure the pipeline:
   Edit `config.py` to enable/disable specific steps:
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

### 3. Run the reconstruction:

**Complete pipeline:**
```bash
python main.py
```

**Single step:**
```bash
python main.py --step run_local_sfm
python main.py --step global_calibration
python main.py --step recalculate_positions
python main.py --step point_cloud_generation
python main.py --step rectification
python main.py --step manual_roi_selection
python main.py --step high_res_rectification
```

**List available steps:**
```bash
python main.py --list-steps
```

**Check step dependencies:**
```bash
python main.py --check-deps rectification
```

### 4. Check results:
   - `outputs/intermediate/`: Intermediate results from each step
   - `outputs/rectified/`: Low resolution rectified images and overviews
   - `outputs/high_res_rectified/`: High resolution rectified images
   - `outputs/roi_selections/`: ROI visualization images
   - `outputs/point_clouds/`: Point cloud visualizations (optional)

## Output Files

### Intermediate Results
- `step1_local_sfm_results.json`: Local reconstructions and calibrations
- `step2_global_calibration_results.json`: Global camera parameters
- `step3_recalculate_positions_results.json`: Global reconstructions
- `step4_point_cloud_results.json`: Point cloud data
- `step5_rectification_results.json`: Rectification data
- `step6_roi_selection_results.json`: ROI selections
- `step7_high_res_rectification_results.json`: High resolution data

### Rectified Images
- `{painting_name}_overview.jpg`: Low resolution overview
- `{painting_name}_rectified_{i}.jpg`: Low resolution rectified images
- `{painting_name}_high_res_{i}.jpg`: High resolution rectified images

### ROI Selections
- `{painting_name}_roi_visualization.jpg`: ROI visualization with selected regions

## Step Independence

Each step can run independently:
- **Input Loading**: Steps load required inputs from intermediate directory
- **Output Storage**: Steps save results to intermediate directory  
- **Dependency Checking**: Steps verify required inputs before execution
- **Resume Capability**: Steps can resume from existing intermediate results

## Testing Individual Steps

Each step can be tested independently:
```bash
python step1_local_sfm.py
python step2_global_calibration.py
python step3_recalculate_positions.py
python step4_point_cloud_generation.py
python step5_rectification.py
python step6_manual_roi_selection.py
python step7_high_res_rectification.py
```

## Algorithm Overview

1. **Local SfM**: Reconstructs 3D camera positions and sparse point cloud for each painting
2. **Global Calibration**: Estimates global camera parameters from all painting sets
3. **Position Recalculation**: Recalculates camera positions using global calibration
4. **Point Cloud Generation**: Generates point clouds and finds painting planes
5. **Low Resolution Rectification**: Warps images to remove perspective distortion
6. **ROI Selection**: Manual selection of regions of interest
7. **High Resolution Rectification**: High-resolution orthorectified images for ROI

## Tips for Best Results

1. **Lighting**: Use diffuse lighting to minimize reflections
2. **Camera angles**: Vary angles by 15-30° for good coverage
3. **Overlap**: Ensure 60-80% overlap between consecutive images
4. **Stability**: Use a tripod or stable surface
5. **Focus**: Ensure all images are in focus
6. **Exposure**: Use consistent exposure settings

## Troubleshooting

### COLMAP fails to reconstruct
- Check that images have sufficient overlap
- Ensure images are not too blurry or underexposed
- Try reducing image resolution if memory is limited

### Poor rectification quality
- Verify that the painting is roughly planar
- Check that camera positions are well-distributed
- Ensure sufficient 3D points are reconstructed

### Steps fail due to missing inputs
- Check that previous steps have completed successfully
- Verify that intermediate results exist in `outputs/intermediate/`
- Run steps in the correct order or use the complete pipeline

## Advanced Usage

### Custom Parameters
Modify the configuration in `config.py` to adjust:
- Grid resolution for rectification
- RANSAC parameters for plane fitting
- Filtering parameters for reflection removal
- Fusion algorithms for super-resolution

### Adding New Steps
1. Create a new step module following the `StepBase` interface
2. Implement the required methods: `run()`, `get_input_requirements()`, `get_output_files()`
3. Add the step to the pipeline in `main.py`
4. Update dependencies and configuration

## Dependencies

- **pycolmap**: COLMAP Python bindings for SfM
- **opencv-python**: Computer vision operations
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Visualization (optional)

## License

This project is open source. Feel free to modify and distribute according to your needs. 