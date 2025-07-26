# Painting Reconstruction System

This system reconstructs paintings from multiple photographs taken from different angles, removes perspective distortion, and creates high-quality orthophotos with reflection removal and super-resolution fusion.

## Features

- **Multi-painting support**: Process multiple painting sets simultaneously
- **Automatic camera calibration**: Estimates camera parameters from all image sets
- **Structure-from-Motion**: Uses COLMAP for robust 3D reconstruction
- **Perspective correction**: Removes camera distortion and perspective effects
- **Reflection removal**: Advanced filtering to reduce reflections and glare
- **Super-resolution fusion**: Combines multiple images for higher quality results
- **Orthophoto generation**: Creates true orthographic projections of paintings

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
├── main.py          # Main reconstruction script
├── requirements.txt # Python dependencies
└── README.md       # This file
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

1. **Prepare your photos**:
   - Create folders for each painting (1, 2, 3, 4, etc.)
   - Place photos of each painting in its respective folder
   - Use the same camera for all photos
   - Take photos from different angles (15-30° apart recommended)

2. **Run the reconstruction**:
```bash
python main.py
```

3. **Check results**:
   - `outputs/fused/`: Final high-quality orthophotos
   - `outputs/rectified/`: Individual rectified images
   - `outputs/calibration/`: Camera calibration data

## Output Files

### Fused Images
- `{painting_name}_fused.jpg`: Final super-resolution orthophoto
- Combines all images with reflection removal and sharpening

### Rectified Images
- `{painting_name}_rectified_{i}.jpg`: Individual perspective-corrected images
- Each image is warped to the painting plane

### Calibration Data
- `camera_calibration.json`: Camera parameters and reconstruction data
- Contains camera matrix, plane parameters, and processing statistics

## Algorithm Overview

1. **Camera Calibration**: Estimates global camera parameters from all painting sets
2. **Structure-from-Motion**: Reconstructs 3D camera positions and sparse point cloud
3. **Plane Detection**: Finds the painting surface using RANSAC plane fitting
4. **Image Rectification**: Warps images to remove perspective distortion
5. **Reflection Removal**: Applies bilateral filtering to reduce glare
6. **Super-resolution Fusion**: Aligns and combines multiple images for higher quality

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

### Reflections not removed
- The algorithm uses bilateral filtering which may not remove all reflections
- Consider using polarized lighting for better results

## Advanced Usage

### Custom Parameters
Modify the `PaintingReconstructor` class to adjust:
- Grid resolution for rectification
- RANSAC parameters for plane fitting
- Filtering parameters for reflection removal
- Fusion algorithms for super-resolution

### Batch Processing
The system automatically processes all painting sets in the Photos directory. Each subfolder is treated as a separate painting.

## Technical Details

- **Camera Model**: Simple pinhole camera with radial distortion
- **Plane Fitting**: RANSAC-based robust plane estimation
- **Image Warping**: Bilinear interpolation for smooth results
- **Fusion**: Optical flow alignment + weighted averaging
- **Reflection Removal**: LAB color space bilateral filtering

## Dependencies

- **pycolmap**: COLMAP Python bindings for SfM
- **opencv-python**: Computer vision operations
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Visualization (optional)

## License

This project is open source. Feel free to modify and distribute according to your needs. 