"""
Test script to verify camera projection and identify issues with black rectified images
"""

import numpy as np
import cv2
from pathlib import Path
import config
from camera_utils import CameraProjector, PlaneProjector

def test_camera_projection():
    """Test basic camera projection functionality"""
    print("=== Testing Camera Projection ===")
    
    # Test camera parameters (typical phone camera)
    camera_params = {
        'model': 'SIMPLE_RADIAL',
        'width': 4032,
        'height': 3024,
        'params': [2000, 2016, 1512, 0.0]  # fx, cx, cy, k1
    }
    
    camera_projector = CameraProjector(camera_params)
    
    # Test projection of some 3D points
    test_points = [
        np.array([0, 0, 1]),      # Point in front of camera
        np.array([1, 0, 1]),      # Point to the right
        np.array([0, 1, 1]),      # Point below
        np.array([-1, 0, 1]),     # Point to the left
        np.array([0, -1, 1]),     # Point above
    ]
    
    print("\nTesting 3D to 2D projection:")
    for i, point_3d in enumerate(test_points):
        point_2d = camera_projector.project_point(point_3d)
        if point_2d is not None:
            print(f"  Point {i}: {point_3d} -> {point_2d}")
        else:
            print(f"  Point {i}: {point_3d} -> FAILED")
    
    # Test unprojection of image corners
    print("\nTesting 2D to 3D unprojection (image corners):")
    corners_2d = [
        [0, 0],           # Top-left
        [camera_params['width'], 0],  # Top-right
        [camera_params['width'], camera_params['height']],  # Bottom-right
        [0, camera_params['height']]  # Bottom-left
    ]
    
    for i, corner_2d in enumerate(corners_2d):
        ray = camera_projector.unproject_ray(np.array(corner_2d))
        if ray is not None:
            print(f"  Corner {i}: {corner_2d} -> ray {ray}")
        else:
            print(f"  Corner {i}: {corner_2d} -> FAILED")

def test_plane_projection():
    """Test plane projection functionality"""
    print("\n=== Testing Plane Projection ===")
    
    # Test plane parameters
    plane_normal = np.array([0, 0, 1])  # Z-up plane
    plane_center = np.array([0, 0, 0])  # Origin
    
    plane_projector = PlaneProjector(plane_normal, plane_center)
    
    # Test some 3D points
    test_points_3d = [
        np.array([1, 0, 0]),   # Point on plane
        np.array([0, 1, 0]),   # Point on plane
        np.array([1, 1, 0]),   # Point on plane
        np.array([0, 0, 1]),   # Point above plane
    ]
    
    print("\nTesting 3D world to 2D plane projection:")
    for i, point_3d in enumerate(test_points_3d):
        point_2d = plane_projector.world_to_plane_2d(point_3d)
        print(f"  Point {i}: {point_3d} -> {point_2d}")
    
    # Test reverse projection
    print("\nTesting 2D plane to 3D world projection:")
    test_points_2d = [
        np.array([1, 0]),   # Point on plane
        np.array([0, 1]),   # Point on plane
        np.array([1, 1]),   # Point on plane
    ]
    
    for i, point_2d in enumerate(test_points_2d):
        point_3d = plane_projector.plane_2d_to_world(point_2d)
        print(f"  Point {i}: {point_2d} -> {point_3d}")

def test_rectification_grid():
    """Test rectification grid creation"""
    print("\n=== Testing Rectification Grid ===")
    
    from camera_utils import create_rectification_grid
    
    # Create test camera projector
    camera_params = {
        'model': 'SIMPLE_RADIAL',
        'width': 4032,
        'height': 3024,
        'params': [2000, 2016, 1512, 0.0]
    }
    camera_projector = CameraProjector(camera_params)
    
    # Create test plane projector
    plane_normal = np.array([0, 0, 1])
    plane_center = np.array([0, 0, 0])
    plane_projector = PlaneProjector(plane_normal, plane_center)
    
    # Create test pose (camera looking at origin from positive Z, 2 units away)
    # Camera is positioned at (0, 0, 2) looking at (0, 0, 0)
    camera_position = np.array([0, 0, 2])
    target_position = np.array([0, 0, 0])
    
    # Calculate rotation matrix (camera looking at target)
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # Create a right-handed coordinate system
    right = np.array([1, 0, 0])  # Assume camera is upright
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    right = np.cross(up, forward)
    
    R = np.array([right, up, forward]).T  # Transpose for rotation matrix
    t = -R @ camera_position
    
    pose = {
        'rotation_matrix': R,
        'translation': t
    }
    
    print(f"Camera position: {camera_position}")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation: {t}")
    
    # Get image corners
    corners_3d = camera_projector.get_image_corners_3d(pose, depth=1.0)
    print(f"Image corners 3D: {[c.tolist() for c in corners_3d]}")
    
    # Create rectification grid
    grid_points_3d, grid_bounds = create_rectification_grid(
        plane_projector, corners_3d, grid_size=64, margin=0.1
    )
    
    print(f"Grid created: {len(grid_points_3d)} points")
    print(f"Grid bounds: {grid_bounds}")
    
    # Test projection of grid points back to image
    print("\nTesting grid point projections:")
    valid_projections = 0
    for i in range(0, len(grid_points_3d), len(grid_points_3d)//10):  # Test every 10th point
        point_3d_world = grid_points_3d[i]
        point_cam = R @ point_3d_world + t
        point_2d = camera_projector.project_point(point_cam)
        if point_2d is not None:
            valid_projections += 1
            print(f"  Grid point {i}: {point_3d_world} -> {point_2d}")
        else:
            print(f"  Grid point {i}: {point_3d_world} -> FAILED")
    
    print(f"Valid projections: {valid_projections}/{len(grid_points_3d)//10}")

if __name__ == "__main__":
    test_camera_projection()
    test_plane_projection()
    test_rectification_grid() 