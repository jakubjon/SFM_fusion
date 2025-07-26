"""
Camera utilities for true orthorectification with proper distortion handling.
This module provides functions for correct projection and back-projection using
the SIMPLE_RADIAL camera model as implemented in COLMAP.
"""

import numpy as np
import cv2
import pycolmap
from typing import Tuple, List, Optional


class CameraProjector:
    """Handles camera projection with proper distortion model"""
    
    def __init__(self, camera_params: dict):
        """
        Initialize with camera parameters
        
        Args:
            camera_params: Dictionary with 'model', 'width', 'height', 'params'
        """
        self.model = camera_params['model']
        self.width = camera_params['width']
        self.height = camera_params['height']
        self.params = np.array(camera_params['params'])
        
        # Create pycolmap Camera object for proper projection
        self.camera = pycolmap.Camera(
            camera_id=1,
            model=self.model,
            width=self.width,
            height=self.height,
            params=self.params
        )
        print(f"CameraProjector initialized: model={self.model}, params={self.params}")
    
    def project_point(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """
        Project 3D point to image coordinates using proper camera model
        
        Args:
            point_3d: 3D point in world coordinates
            
        Returns:
            2D image coordinates or None if projection fails
        """
        try:
            # For SIMPLE_RADIAL model, manually implement projection
            if self.model == 'SIMPLE_RADIAL':
                fx, cx, cy, k1 = self.params
                
                # Normalize 3D point (assume it's in camera coordinates)
                if point_3d[2] <= 0:
                    return None  # Point behind camera
                
                # Project to normalized coordinates
                x = point_3d[0] / point_3d[2]
                y = point_3d[1] / point_3d[2]
                
                # Apply radial distortion
                r2 = x*x + y*y
                factor = 1 + k1 * r2
                x_dist = x * factor
                y_dist = y * factor
                
                # Apply camera matrix
                u = fx * x_dist + cx
                v = fx * y_dist + cy
                
                point_2d = np.array([u, v])
                
                # Check if point is within image bounds
                if (0 <= point_2d[0] < self.width and 0 <= point_2d[1] < self.height):
                    return point_2d
                else:
                    return None
            else:
                # For pinhole model
                fx, cx, cy = self.params[:3]
                
                if point_3d[2] <= 0:
                    return None  # Point behind camera
                
                # Project to normalized coordinates
                x = point_3d[0] / point_3d[2]
                y = point_3d[1] / point_3d[2]
                
                # Apply camera matrix
                u = fx * x + cx
                v = fx * y + cy
                
                point_2d = np.array([u, v])
                
                # Check if point is within image bounds
                if (0 <= point_2d[0] < self.width and 0 <= point_2d[1] < self.height):
                    return point_2d
                else:
                    return None
                    
        except Exception as e:
            print(f"Projection failed for point {point_3d}: {e}")
            return None
    
    def unproject_ray(self, point_2d: np.ndarray) -> Optional[np.ndarray]:
        """
        Unproject 2D image point to ray direction (for back-projection)
        
        Args:
            point_2d: 2D image coordinates
            
        Returns:
            Ray direction vector (normalized) or None if unprojection fails
        """
        try:
            # For SIMPLE_RADIAL model, we need to handle distortion
            if self.model == 'SIMPLE_RADIAL':
                fx, cx, cy, k1 = self.params
                
                # Convert to normalized coordinates
                x = (point_2d[0] - cx) / fx
                y = (point_2d[1] - cy) / fx
                
                # Apply radial distortion correction
                r2 = x*x + y*y
                r4 = r2*r2
                
                # For back-projection, we need to solve the distortion equation
                # This is an approximation - for exact solution, iterative method would be needed
                # For small distortion, this works well
                if abs(k1) < 0.1:  # Small distortion approximation
                    x_undist = x / (1 + k1 * r2)
                    y_undist = y / (1 + k1 * r2)
                else:
                    # For larger distortion, use iterative approach
                    x_undist, y_undist = self._iterative_undistort(x, y, k1)
                
                # Create ray direction
                ray = np.array([x_undist, y_undist, 1.0])
                return ray / np.linalg.norm(ray)
            else:
                # For pinhole model
                fx, cx, cy = self.params[:3]
                x = (point_2d[0] - cx) / fx
                y = (point_2d[1] - cy) / fx
                ray = np.array([x, y, 1.0])
                return ray / np.linalg.norm(ray)
                
        except Exception as e:
            print(f"Unprojection failed: {e}")
            return None
    
    def _iterative_undistort(self, x_dist: float, y_dist: float, k1: float, 
                           max_iterations: int = 10, tolerance: float = 1e-6) -> Tuple[float, float]:
        """
        Iteratively solve for undistorted coordinates
        
        Args:
            x_dist, y_dist: Distorted coordinates
            k1: Radial distortion coefficient
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Undistorted coordinates (x_undist, y_undist)
        """
        x_undist = x_dist
        y_undist = y_dist
        
        for _ in range(max_iterations):
            r2 = x_undist*x_undist + y_undist*y_undist
            factor = 1 + k1 * r2
            
            x_new = x_dist / factor
            y_new = y_dist / factor
            
            if (abs(x_new - x_undist) < tolerance and 
                abs(y_new - y_undist) < tolerance):
                break
                
            x_undist = x_new
            y_undist = y_new
        
        return x_undist, y_undist
    
    def get_image_corners_3d(self, pose: dict, plane_projector: 'PlaneProjector') -> List[np.ndarray]:
        """
        Get 3D coordinates of image corners by intersecting rays with painting plane
        
        Args:
            pose: Camera pose with 'rotation_matrix' and 'translation'
            plane_projector: Plane projector for plane intersection
            
        Returns:
            List of 4 corner points in 3D world coordinates (intersection with plane)
        """
        # Image corners in image coordinates
        corners_2d = np.array([
            [0, 0],           # Top-left
            [self.width, 0],  # Top-right
            [self.width, self.height],  # Bottom-right
            [0, self.height]  # Bottom-left
        ])
        
        # Get camera center and rotation
        R = np.array(pose['rotation_matrix'])
        t = np.array(pose['translation'])
        camera_center = -R.T @ t
        
        corners_3d = []
        for corner_2d in corners_2d:
            # Step 1: Remove distortion (normalize image)
            normalized_ray = self.unproject_ray(corner_2d)
            if normalized_ray is not None:
                # Step 2: Transform ray to world coordinates
                ray_world = R.T @ normalized_ray
                
                # Step 3: Intersect ray with painting plane
                # Ray equation: P = camera_center + t * ray_world
                # Plane equation: (P - plane_center) · plane_normal = 0
                # Solve for t: (camera_center + t * ray_world - plane_center) · plane_normal = 0
                # t = (plane_center - camera_center) · plane_normal / (ray_world · plane_normal)
                
                plane_normal = plane_projector.plane_normal
                plane_center = plane_projector.plane_center
                
                numerator = np.dot(plane_center - camera_center, plane_normal)
                denominator = np.dot(ray_world, plane_normal)
                
                if abs(denominator) > 1e-6:  # Ray not parallel to plane
                    t_intersection = numerator / denominator
                    if t_intersection > 0:  # Intersection in front of camera
                        corner_3d = camera_center + t_intersection * ray_world
                        corners_3d.append(corner_3d)
                    else:
                        # Ray intersects behind camera, use fallback
                        corner_3d = camera_center + 1.0 * ray_world
                        corners_3d.append(corner_3d)
                else:
                    # Ray parallel to plane, use fallback
                    corner_3d = camera_center + 1.0 * ray_world
                    corners_3d.append(corner_3d)
            else:
                # Fallback: use simple pinhole model
                fx, cx, cy = self.params[:3]
                x = (corner_2d[0] - cx) / fx
                y = (corner_2d[1] - cy) / fx
                ray = np.array([x, y, 1.0])
                ray = ray / np.linalg.norm(ray)
                ray_world = R.T @ ray
                
                # Intersect with plane
                plane_normal = plane_projector.plane_normal
                plane_center = plane_projector.plane_center
                
                numerator = np.dot(plane_center - camera_center, plane_normal)
                denominator = np.dot(ray_world, plane_normal)
                
                if abs(denominator) > 1e-6:
                    t_intersection = numerator / denominator
                    if t_intersection > 0:
                        corner_3d = camera_center + t_intersection * ray_world
                    else:
                        corner_3d = camera_center + 1.0 * ray_world
                else:
                    corner_3d = camera_center + 1.0 * ray_world
                
                corners_3d.append(corner_3d)
        
        return corners_3d


class PlaneProjector:
    """Handles projection between 3D world and 2D painting plane"""
    
    def __init__(self, plane_normal: np.ndarray, plane_center: np.ndarray):
        """
        Initialize with painting plane parameters
        
        Args:
            plane_normal: Normal vector of the painting plane
            plane_center: Center point of the painting plane
        """
        self.plane_normal = plane_normal / np.linalg.norm(plane_normal)
        self.plane_center = plane_center
        
        # Create 2D coordinate system on the plane
        self._create_coordinate_system()
        print(f"PlaneProjector initialized: normal={self.plane_normal}, center={self.plane_center}")
    
    def _create_coordinate_system(self):
        """Create orthonormal basis vectors for the 2D coordinate system"""
        # Find two vectors perpendicular to the normal
        if abs(self.plane_normal[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        v2 = np.cross(self.plane_normal, v1)
        v1 = np.cross(v2, self.plane_normal)
        
        # Normalize
        self.v1 = v1 / np.linalg.norm(v1)
        self.v2 = v2 / np.linalg.norm(v2)
        print(f"Plane coordinate system: v1={self.v1}, v2={self.v2}")
    
    def world_to_plane_2d(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D world point to 2D plane coordinates
        
        Args:
            point_3d: 3D point in world coordinates
            
        Returns:
            2D coordinates on the painting plane
        """
        # Vector from plane center to point
        vec = point_3d - self.plane_center
        
        # Project to 2D coordinate system
        u = np.dot(vec, self.v1)
        v = np.dot(vec, self.v2)
        
        return np.array([u, v])
    
    def plane_2d_to_world(self, point_2d: np.ndarray) -> np.ndarray:
        """
        Convert 2D plane coordinates back to 3D world coordinates
        
        Args:
            point_2d: 2D coordinates on the painting plane [u, v]
            
        Returns:
            3D point in world coordinates
        """
        return self.plane_center + point_2d[0] * self.v1 + point_2d[1] * self.v2


def create_rectification_grid(plane_projector: PlaneProjector, 
                            corners_3d: List[np.ndarray],
                            grid_size: int,
                            margin: float = 0.1,
                            reconstruction_points_3d: Optional[List[np.ndarray]] = None,
                            global_bounds: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, dict]:
    """
    Create a rectification grid on the painting plane that follows the actual painting shape
    
    Args:
        plane_projector: Plane projector instance
        corners_3d: 3D coordinates of image corners
        grid_size: Size of the rectification grid
        margin: Margin around the painting (fraction of painting size)
        reconstruction_points_3d: Optional 3D points from COLMAP reconstruction for better bounds
        global_bounds: Optional global bounds (min_u, max_u, min_v, max_v) from all images
        
    Returns:
        Tuple of (grid_points_3d, grid_bounds)
    """
    # If global bounds are provided, use them directly
    if global_bounds is not None:
        min_u, max_u, min_v, max_v = global_bounds
        print(f"Using provided global bounds: u=[{min_u:.3f}, {max_u:.3f}], v=[{min_v:.3f}, {max_v:.3f}]")
    else:
        # Project corners to 2D plane
        corners_2d = []
        for corner_3d in corners_3d:
            corner_2d = plane_projector.world_to_plane_2d(corner_3d)
            corners_2d.append(corner_2d)
        
        corners_2d = np.array(corners_2d)
        print(f"Corner projections to plane: {corners_2d}")
        
        # Find bounds of corners
        min_u, min_v = corners_2d.min(axis=0)
        max_u, max_v = corners_2d.max(axis=0)
        
        # If we have reconstruction points, use them to get better bounds and shape
        painting_shape_points = None
        if reconstruction_points_3d is not None and len(reconstruction_points_3d) > 0:
            print(f"Using {len(reconstruction_points_3d)} reconstruction points for better bounds and shape")
            
            # Project all reconstruction points to plane
            points_2d = []
            for point_3d in reconstruction_points_3d:
                point_2d = plane_projector.world_to_plane_2d(point_3d)
                points_2d.append(point_2d)
            
            points_2d = np.array(points_2d)
            
            # Find bounds of all points
            all_min_u, all_min_v = points_2d.min(axis=0)
            all_max_u, all_max_v = points_2d.max(axis=0)
            
            # Use the union of corner bounds and reconstruction point bounds
            min_u = min(min_u, all_min_u)
            max_u = max(max_u, all_max_u)
            min_v = min(min_v, all_min_v)
            max_v = max(max_v, all_max_v)
            
            # Store painting shape points for adaptive grid
            painting_shape_points = points_2d
            
            print(f"Combined bounds from corners and reconstruction points: u=[{min_u:.3f}, {max_u:.3f}], v=[{min_v:.3f}, {max_v:.3f}]")
    
    # Add margin
    u_range = max_u - min_u
    v_range = max_v - min_v
    min_u -= u_range * margin
    max_u += u_range * margin
    min_v -= v_range * margin
    max_v += v_range * margin
    
    print(f"Final grid bounds: u=[{min_u:.3f}, {max_u:.3f}], v=[{min_v:.3f}, {max_v:.3f}]")
    
    # Create adaptive grid that follows painting shape
    grid_points_3d = []
    valid_grid_points = 0
    
    # If we have painting shape points, create a more efficient mask
    painting_mask = None
    if reconstruction_points_3d is not None and len(reconstruction_points_3d) > 10:
        # Create a coarse mask first to speed up processing
        mask_size = 100  # Coarse mask size
        mask = np.zeros((mask_size, mask_size), dtype=bool)
        
        # Project reconstruction points to mask coordinates
        for point_3d in reconstruction_points_3d:
            point_2d = plane_projector.world_to_plane_2d(point_3d)
            mask_i = int((point_2d[0] - min_u) / (max_u - min_u) * (mask_size - 1))
            mask_j = int((point_2d[1] - min_v) / (max_v - min_v) * (mask_size - 1))
            if 0 <= mask_i < mask_size and 0 <= mask_j < mask_size:
                mask[mask_j, mask_i] = True
        
        # Dilate the mask to include nearby areas
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=2)
        painting_mask = mask
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Map grid coordinates to 2D plane coordinates
            u = min_u + (i / (grid_size - 1)) * (max_u - min_u)
            v = min_v + (j / (grid_size - 1)) * (max_v - min_v)
            
            # Check if this point is within the painting area using coarse mask
            is_inside_painting = True
            if painting_mask is not None:
                # Map to mask coordinates
                mask_i = int((u - min_u) / (max_u - min_u) * (mask_size - 1))
                mask_j = int((v - min_v) / (max_v - min_v) * (mask_size - 1))
                
                if 0 <= mask_i < mask_size and 0 <= mask_j < mask_size:
                    is_inside_painting = painting_mask[mask_j, mask_i]
                else:
                    is_inside_painting = False
            
            if is_inside_painting:
                # Convert back to 3D
                point_3d = plane_projector.plane_2d_to_world(np.array([u, v]))
                grid_points_3d.append(point_3d)
                valid_grid_points += 1
            else:
                # Add a point far away (will be filtered out during projection)
                grid_points_3d.append(np.array([0, 0, -1000]))  # Far behind camera
    
    grid_points_3d = np.array(grid_points_3d)
    
    print(f"Created adaptive grid: {valid_grid_points}/{grid_size*grid_size} points within painting area")
    
    grid_bounds = {
        'min_u': min_u, 'max_u': max_u,
        'min_v': min_v, 'max_v': max_v,
        'grid_size': grid_size,
        'valid_points': valid_grid_points
    }
    
    return grid_points_3d, grid_bounds


def rectify_image_true_ortho_global(image: np.ndarray, 
                                   camera_projector: CameraProjector,
                                   pose: dict,
                                   grid_points_3d: np.ndarray,
                                   grid_size: int) -> np.ndarray:
    """
    Perform true orthorectification using global grid (same grid for all images of painting)
    
    Args:
        image: Input image
        camera_projector: Camera projector instance
        pose: Camera pose
        grid_points_3d: 3D grid points on painting plane (in world coordinates)
        grid_size: Size of the rectification grid
        
    Returns:
        Rectified image
    """
    # Get camera pose
    R = np.array(pose['rotation_matrix'])
    t = np.array(pose['translation'])
    
    # Create rectified image
    rectified = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    # Debug counters
    valid_projections = 0
    valid_samples = 0
    
    # For each grid point, project to image and sample
    for i, point_3d_world in enumerate(grid_points_3d):
        # Skip points that are far behind camera (invalid points from adaptive grid)
        if point_3d_world[2] < -100:
            continue
            
        # Transform point from world coordinates to camera coordinates
        point_cam = R @ point_3d_world + t
        
        # Project to image coordinates
        point_2d = camera_projector.project_point(point_cam)
        
        if point_2d is not None:
            valid_projections += 1
            
            # Sample image with bilinear interpolation
            x, y = point_2d
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
            
            # Bilinear interpolation weights
            wx = x - x0
            wy = y - y0
            
            # Sample from image
            if (0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0] and
                0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]):
                
                c00 = image[y0, x0].astype(np.float32)
                c01 = image[y0, x1].astype(np.float32)
                c10 = image[y1, x0].astype(np.float32)
                c11 = image[y1, x1].astype(np.float32)
                
                # Interpolate
                c0 = c00 * (1 - wx) + c01 * wx
                c1 = c10 * (1 - wx) + c11 * wx
                color = c0 * (1 - wy) + c1 * wy
                
                # Set pixel in rectified image
                grid_i = i % grid_size
                grid_j = i // grid_size
                rectified[grid_j, grid_i] = color.astype(np.uint8)
                valid_samples += 1
    
    print(f"Rectification stats: {valid_projections}/{len(grid_points_3d)} valid projections, {valid_samples} valid samples")
    
    return rectified


def create_overview_fusion(rectified_images: List[np.ndarray]) -> np.ndarray:
    """
    Create overview by fusing multiple rectified images with proper alignment
    
    Args:
        rectified_images: List of rectified images
        
    Returns:
        Fused overview image
    """
    if not rectified_images:
        return None
    
    if len(rectified_images) == 1:
        return rectified_images[0]
    
    # Convert to float for processing
    images_float = [img.astype(np.float32) for img in rectified_images]
    
    # Create a mask to focus on areas with actual content
    # This helps avoid averaging in areas with no painting content
    mask = np.zeros_like(images_float[0][:, :, 0], dtype=np.float32)
    
    for img in images_float:
        # Create a simple mask based on image intensity
        intensity = np.mean(img, axis=2)
        # Areas with low intensity are likely background
        img_mask = (intensity > 10).astype(np.float32)
        mask += img_mask
    
    # Normalize mask
    mask = np.clip(mask, 0, 1)
    
    # Simple weighted average with mask
    weights = []
    for img in images_float:
        # Calculate variance as quality measure
        variance = np.var(img)
        weights.append(variance)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Weighted average with mask
    overview = np.zeros_like(images_float[0])
    for img, weight in zip(images_float, weights):
        overview += weight * img
    
    # Apply mask to focus on painting areas
    for c in range(3):
        overview[:, :, c] *= mask
    
    # Convert back to uint8
    overview = np.clip(overview, 0, 255).astype(np.uint8)
    
    return overview 