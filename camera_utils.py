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


def create_simple_envelope_grid(plane_projector: PlaneProjector, 
                               all_corners_2d: List[np.ndarray],
                               grid_size: int) -> Tuple[np.ndarray, dict]:
    """
    Create a simple rectangular grid based on the envelope of all image corner projections
    
    Args:
        plane_projector: Plane projector instance
        all_corners_2d: List of 2D corner projections from all images
        grid_size: Base size of the rectification grid (will be adjusted for aspect ratio)
        margin: Margin around the envelope (fraction of envelope size) - only used if add_margin=True
        add_margin: Whether to add margin around the envelope (default: False for tight fit)
        
    Returns:
        Tuple of (grid_points_3d, grid_bounds)
    """
    if len(all_corners_2d) == 0:
        print("[ERROR] No corner projections provided")
        return None, None
    
    # Convert to numpy array
    all_corners_2d = np.array(all_corners_2d)
    
    # Simple envelope: min/max of all corners
    min_u, min_v = all_corners_2d.min(axis=0)
    max_u, max_v = all_corners_2d.max(axis=0)
    
    print(f"Simple envelope from all corners: u=[{min_u:.3f}, {max_u:.3f}], v=[{min_v:.3f}, {max_v:.3f}]")
    
    
    # Calculate aspect ratio and adjust grid size
    u_range = max_u - min_u
    v_range = max_v - min_v
    aspect_ratio = u_range / v_range
    
    if aspect_ratio > 1:
        # Wider than tall - use grid_size for width
        grid_width = grid_size
        grid_height = int(grid_size / aspect_ratio)
    else:
        # Taller than wide - use grid_size for height
        grid_height = grid_size
        grid_width = int(grid_size * aspect_ratio)
    
    print(f"Aspect ratio: {aspect_ratio:.3f}, Grid size: {grid_width}x{grid_height}")
    
    # Create rectangular grid respecting aspect ratio
    grid_points_3d = []
    u_coords = np.linspace(min_u, max_u, grid_width)
    v_coords = np.linspace(min_v, max_v, grid_height)
    
    for i, v in enumerate(v_coords):
        for j, u in enumerate(u_coords):
            point_3d = plane_projector.plane_2d_to_world(np.array([u, v]))
            grid_points_3d.append(point_3d)
    
    grid_bounds = {
        'min_u': min_u,
        'max_u': max_u,
        'min_v': min_v,
        'max_v': max_v,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'aspect_ratio': aspect_ratio,
        'valid_points': len(grid_points_3d)
    }
    
    print(f"Created aspect-ratio-corrected grid: {len(grid_points_3d)} points ({grid_width}x{grid_height})")
    return np.array(grid_points_3d), grid_bounds


def rectify_image_true_ortho_global(image: np.ndarray, 
                                   camera_projector: CameraProjector,
                                   pose: dict,
                                   grid_points_3d: np.ndarray,
                                   grid_width: int,
                                   grid_height: int) -> np.ndarray:
    """
    Perform true orthorectification using global grid (same grid for all images of painting)
    
    Args:
        image: Input image
        camera_projector: Camera projector instance
        pose: Camera pose
        grid_points_3d: 3D grid points on painting plane (in world coordinates)
        grid_width: Width of the rectification grid
        grid_height: Height of the rectification grid
        
    Returns:
        Rectified image
    """
    # Get camera pose
    R = np.array(pose['rotation_matrix'])
    t = np.array(pose['translation'])
    
    # Create rectified image with correct aspect ratio
    rectified = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
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
                grid_i = i % grid_width
                grid_j = i // grid_width
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

