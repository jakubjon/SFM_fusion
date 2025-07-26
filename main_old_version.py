import os
import pycolmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import ndimage
from sklearn.cluster import DBSCAN
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')
import config

def log_step(step_name, painting_name=None):
    """Log a step with timestamp and painting name if applicable"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if painting_name:
        print(f"\n## {timestamp} - {step_name} - Painting: {painting_name}")
    else:
        print(f"\n## {timestamp} - {step_name}")
    print("=" * 80)

class PaintingReconstructor:
    def __init__(self, photos_dir=config.PHOTOS_DIR, output_dir=config.OUTPUT_DIR):
        self.photos_dir = Path(photos_dir)
        self.output_dir = Path(output_dir)
        self.rectified_dir = self.output_dir / 'rectified'
        self.fused_dir = self.output_dir / 'fused'
        self.calibration_dir = self.output_dir / 'calibration'
        
        # Create output directories
        for dir_path in [self.output_dir, self.rectified_dir, self.fused_dir, self.calibration_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Get painting sets
        self.painting_sets = [d for d in self.photos_dir.iterdir() if d.is_dir()]
        print(f"Found {len(self.painting_sets)} painting sets: {[p.name for p in self.painting_sets]}")
        
        # Global camera calibration (will be refined)
        self.global_camera_params = None
        
        # Store local camera calibrations for comparison
        self.local_camera_calibrations = {}
        
    def extract_exif_calibration(self, image_path):
        """Extract preliminary camera calibration from EXIF data"""
        try:
            import exifread
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
            
            # Extract focal length and sensor info
            focal_length = None
            if 'EXIF FocalLength' in tags:
                focal_length = float(tags['EXIF FocalLength'].values[0])
            
            # Get image dimensions
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]
                
                # Estimate principal point (center of image)
                cx, cy = width / 2, height / 2
                
                if focal_length:
                    return {
                        'width': width,
                        'height': height,
                        'params': [focal_length, cx, cy, 0, 0],  # fx, cx, cy, k1, k2
                        'model': 'SIMPLE_PINHOLE'
                    }
                else:
                    return {
                        'width': width,
                        'height': height,
                        'params': [max(width, height), cx, cy, 0, 0],  # Default focal length
                        'model': 'SIMPLE_PINHOLE'
                    }
        except ImportError:
            print("exifread not available, using default calibration")
        except Exception as e:
            print(f"Error extracting EXIF: {e}")
        
        return None
        
    def run_colmap_sfm(self, image_dir, output_path, database_path, painting_name):
        """Run COLMAP SfM on a set of images with detailed logging"""
        log_step("Registration", painting_name)
        
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create database
            log_step("Creating database", painting_name)
            database = pycolmap.Database(str(database_path))
            print(f"Database created: {database_path}")
            
            # Import images to database
            log_step("Importing images", painting_name)
            pycolmap.import_images(str(database_path), str(image_dir))
            print(f"Images imported to database")
            
            # Extract features
            log_step("Feature extraction", painting_name)
            pycolmap.extract_features(str(database_path), str(image_dir))
            print(f"Features extracted")
            
            # Match features
            log_step("Feature matching", painting_name)
            pycolmap.match_exhaustive(str(database_path))
            print(f"Features matched")
            
            # Run incremental mapping
            log_step("Preliminary camera position determination", painting_name)
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(image_dir),
                output_path=str(output_path)
            )
            print(f"Initial reconstruction completed")
            
            # Return the first (and usually only) reconstruction
            if reconstructions:
                recon = list(reconstructions.values())[0]
                
                # Calculate reprojection error
                if hasattr(recon, 'points3D') and recon.points3D:
                    total_error = 0
                    total_points = 0
                    for point3D in recon.points3D.values():
                        if hasattr(point3D, 'error'):
                            total_error += point3D.error
                            total_points += 1
                    
                    if total_points > 0:
                        avg_error = total_error / total_points
                        print(f"Average reprojection error: {avg_error:.4f} pixels")
                
                return recon
            return None
        except Exception as e:
            print(f"COLMAP failed for {image_dir}: {e}")
            return None
    
    def run_local_sfm_only(self, painting_set_path, painting_name):
        """Run only local SfM without rectification (for initial camera positions)"""
        log_step("Local SfM (initial camera positions)", painting_name)
        
        # Ensure output directory exists
        output_dir = self.output_dir / painting_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run COLMAP SfM
        recon = self.run_colmap_sfm(
            painting_set_path, 
            output_dir,
            output_dir / 'database.db',
            painting_name
        )
        
        if not recon:
            print(f"❌ COLMAP reconstruction failed for {painting_name}")
            return None
        
        # Store local camera calibration
        if recon.cameras:
            ref_camera_id = list(recon.cameras.keys())[0]
            camera = recon.cameras[ref_camera_id]
            
            # Convert to pinhole model for local calibration
            local_params = camera.params.tolist() if hasattr(camera.params, 'tolist') else list(camera.params)
            if len(local_params) > 3:
                # Keep only first 3 parameters for pinhole model
                local_params = local_params[:3]
            
            self.local_camera_calibrations[painting_name] = {
                'width': camera.width,
                'height': camera.height,
                'params': local_params,
                'model': 'SIMPLE_PINHOLE'  # Use pinhole for local calibrations
            }
            print(f"Local camera calibration stored for {painting_name} (pinhole model)")
        
        return recon
    
    def estimate_global_camera_calibration(self):
        """Estimate global camera calibration from all painting sets"""
        log_step("Preliminary global camera intrinsic determination")
        
        # Try to get calibration from first image's EXIF
        first_image = None
        for painting_set in self.painting_sets:
            image_files = list(painting_set.glob('*.jpg')) + list(painting_set.glob('*.jpeg')) + list(painting_set.glob('*.png'))
            if image_files:
                first_image = image_files[0]
                break
        
        if first_image:
            exif_calibration = self.extract_exif_calibration(first_image)
            if exif_calibration:
                print(f"EXIF calibration found: focal_length={exif_calibration['params'][0]:.2f}")
                # Convert to more complex model for global calibration
                exif_calibration['model'] = 'SIMPLE_RADIAL'
                # Add radial distortion parameter
                if len(exif_calibration['params']) == 3:
                    exif_calibration['params'].append(0.0)  # Add k1 parameter
                return exif_calibration
        
        print("No EXIF calibration found, using default parameters with SIMPLE_RADIAL model")
        default_params = config.DEFAULT_CAMERA_PARAMS.copy()
        default_params['model'] = 'SIMPLE_RADIAL'
        if len(default_params['params']) == 3:
            default_params['params'].append(0.0)  # Add k1 parameter
        return default_params
    
    def process_painting_set(self, painting_set_path, painting_name):
        """Process a single painting set with detailed logging"""
        log_step("Processing painting set", painting_name)
        
        # Ensure output directory exists
        output_dir = self.output_dir / painting_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run COLMAP SfM
        recon = self.run_colmap_sfm(
            painting_set_path, 
            output_dir,
            output_dir / 'database.db',
            painting_name
        )
        
        if not recon:
            print(f"❌ COLMAP reconstruction failed for {painting_name}")
            return None
        
        # Store local camera calibration
        if recon.cameras:
            ref_camera_id = list(recon.cameras.keys())[0]
            camera = recon.cameras[ref_camera_id]
            
            # Convert to pinhole model for local calibration
            local_params = camera.params.tolist() if hasattr(camera.params, 'tolist') else list(camera.params)
            if len(local_params) > 3:
                # Keep only first 3 parameters for pinhole model
                local_params = local_params[:3]
            
            self.local_camera_calibrations[painting_name] = {
                'width': camera.width,
                'height': camera.height,
                'params': local_params,
                'model': 'SIMPLE_PINHOLE'  # Use pinhole for local calibrations
            }
            print(f"Local camera calibration stored for {painting_name} (pinhole model)")
        
        # Find the painting plane
        log_step("Point cloud generation", painting_name)
        plane_normal, plane_center = self.find_painting_plane(recon, None)
        
        # Get camera parameters
        if self.global_camera_params:
            camera_params = self.global_camera_params
        else:
            # Use camera from reconstruction
            ref_camera_id = list(recon.cameras.keys())[0]
            camera_params = recon.cameras[ref_camera_id]
        
        # Extract camera intrinsics
        if hasattr(camera_params, 'calibration_matrix'):
            K = camera_params.calibration_matrix()
        elif isinstance(camera_params, dict):
            # Create calibration matrix from parameters
            params = camera_params['params']
            if len(params) >= 3:
                fx, cx, cy = params[:3]
                K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
            else:
                print(f"Invalid camera parameters for {painting_name}")
                return None
        else:
            # Handle pycolmap Camera object
            params = camera_params.params
            if len(params) >= 3:
                fx, cx, cy = params[:3]
                K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
            else:
                print(f"Invalid camera parameters for {painting_name}")
                return None
        
        # Rectify all images
        log_step("Painting plane determination", painting_name)
        rectified_images = []
        image_files = list(painting_set_path.glob('*.jpg')) + list(painting_set_path.glob('*.jpeg')) + list(painting_set_path.glob('*.png'))
        
        for i, image_file in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Create a simple image object for rectification
            class SimpleImage:
                def __init__(self, name):
                    self.name = name
                    self.image_id = name
            
            image = SimpleImage(image_file.name)
            
            rectified = self.rectify_image_to_plane(
                image, recon, K, plane_normal, plane_center, painting_set_path
            )
            
            if rectified is not None:
                rectified_images.append(rectified)
                # Save rectified image
                output_path = self.rectified_dir / f"{painting_name}_{image_file.stem}_rectified.jpg"
                cv2.imwrite(str(output_path), rectified)
                print(f"Saved rectified image: {output_path.name}")
        
        if not rectified_images:
            print(f"❌ No images could be rectified for {painting_name}")
            return None
        
        # Create overview (highly reduced orthorectified pictures)
        log_step("Creating painting overview", painting_name)
        if rectified_images:
            # Create a simple overview by averaging all rectified images
            overview = np.mean(rectified_images, axis=0).astype(np.uint8)
            overview_path = self.rectified_dir / f"{painting_name}_overview.jpg"
            cv2.imwrite(str(overview_path), overview)
            print(f"Overview saved: {overview_path.name}")
        
        # Skip rest of processing for now (as requested)
        print("Skipping advanced post-processing (commented out)")
        
        return {
            'painting_name': painting_name,
            'num_images': len(rectified_images),
            'plane_normal': plane_normal.tolist(),
            'plane_center': plane_center.tolist(),
            'camera_params': self.to_serializable(camera_params)
        }
    
    def process_painting_with_global_calibration(self, local_recon, painting_name):
        """Process painting with global calibration (after recalculation)"""
        log_step("Processing painting with global calibration", painting_name)
        
        # Find the painting plane using global calibration
        log_step("Point cloud generation with global calibration", painting_name)
        plane_normal, plane_center = self.find_painting_plane(local_recon, None)
        
        # Use global camera parameters
        if not self.global_camera_params:
            print(f"No global camera parameters available for {painting_name}")
            return None
        
        camera_params = self.global_camera_params
        
        # Extract camera intrinsics from global calibration
        if isinstance(camera_params, dict):
            params = camera_params['params']
            if len(params) >= 4:  # SIMPLE_RADIAL has 4 parameters
                fx, cx, cy, k1 = params[:4]
                K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
                print(f"Using global calibration: focal_length={fx:.2f}, k1={k1:.6f}")
            else:
                print(f"Invalid global camera parameters for {painting_name}")
                return None
        else:
            print(f"Invalid global camera parameters format for {painting_name}")
            return None
        
        # Rectify all images with global calibration
        log_step("Painting plane determination with global calibration", painting_name)
        rectified_images = []
        
        # Get the original painting set path
        painting_set_path = self.photos_dir / painting_name
        image_files = list(painting_set_path.glob('*.jpg')) + list(painting_set_path.glob('*.jpeg')) + list(painting_set_path.glob('*.png'))
        
        for i, image_file in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Create a simple image object for rectification
            class SimpleImage:
                def __init__(self, name):
                    self.name = name
                    self.image_id = name
            
            image = SimpleImage(image_file.name)
            
            rectified = self.rectify_image_to_plane(
                image, local_recon, K, plane_normal, plane_center, painting_set_path
            )
            
            if rectified is not None:
                rectified_images.append(rectified)
                # Save rectified image
                output_path = self.rectified_dir / f"{painting_name}_{image_file.stem}_rectified.jpg"
                cv2.imwrite(str(output_path), rectified)
                print(f"Saved rectified image: {output_path.name}")
        
        if not rectified_images:
            print(f"❌ No images could be rectified for {painting_name}")
            return None
        
        # Create overview (highly reduced orthorectified pictures)
        log_step("Creating painting overview with global calibration", painting_name)
        if rectified_images:
            # Create a simple overview by averaging all rectified images
            overview = np.mean(rectified_images, axis=0).astype(np.uint8)
            
            # Add original frame outlines if enabled
            if config.RECTIFICATION_CONFIG['show_original_frames']:
                overview = self.add_frame_outlines_to_overview(overview, painting_name)
            
            overview_path = self.rectified_dir / f"{painting_name}_overview.jpg"
            cv2.imwrite(str(overview_path), overview)
            print(f"Overview saved: {overview_path.name}")
            
            # Save overview data
            if config.INTERMEDIATE_RESULTS['save_overview_data']:
                self.save_intermediate_result(f"overview_data_{painting_name}", {
                    'num_images': len(rectified_images),
                    'overview_size': overview.shape,
                    'has_frame_outlines': config.RECTIFICATION_CONFIG['show_original_frames'],
                    'painting_name': painting_name,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Skip rest of processing for now (as requested)
        print("Skipping advanced post-processing (commented out)")
        
        return {
            'painting_name': painting_name,
            'num_images': len(rectified_images),
            'plane_normal': plane_normal.tolist(),
            'plane_center': plane_center.tolist(),
            'camera_params': self.to_serializable(camera_params)
        }
    
    def find_painting_plane(self, recon, ref_image):
        """Find the painting plane using RANSAC"""
        print("Finding painting plane...")
        
        # Get 3D points from the reconstruction
        points3D = []
        for point3D in recon.points3D.values():
            points3D.append(point3D.xyz)
        
        if len(points3D) < 10:
            print("Not enough 3D points for plane fitting")
            return np.array([0, 0, 1]), np.array([0, 0, 0])
        
        points3D = np.array(points3D)
        print(f"Using {len(points3D)} 3D points for plane fitting")

        # Use RANSAC to find the dominant plane
        best_normal = None
        best_center = None
        max_inliers = 0
        
        for _ in range(config.RANSAC_ITERATIONS):
            # Sample 3 points
            idx = np.random.choice(len(points3D), 3, replace=False)
            p1, p2, p3 = points3D[idx]
            
            # Calculate plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # Calculate distance to plane
            center = (p1 + p2 + p3) / 3
            distances = np.abs(np.dot(points3D - center, normal))
            
            # Count inliers
            inliers = np.sum(distances < config.PLANE_THRESHOLD)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_normal = normal
                best_center = center
        
        print(f"Found plane with {max_inliers} inliers")
        return best_normal, best_center
    
    def qvec2rotmat(self, qvec):
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def rectify_image_to_plane(self, image, recon, K, plane_normal, plane_center, image_dir):
        """Rectify image to painting plane using proper 2D coordinate system with rectangular envelope"""
        # Get camera pose from the reconstruction
        img_obj = None
        for img_id, img in recon.images.items():
            if img.name == image.name:
                img_obj = img
                break
        
        if img_obj is None:
            print(f"Could not find image {image.name} in reconstruction")
            return None
        
        # Try different ways to get the pose
        pose = None
        
        # Method 1: Try cam_from_world (this is the correct one!)
        if hasattr(img_obj, 'cam_from_world'):
            rigid = img_obj.cam_from_world()
            # Convert Rigid3d to transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = np.array(rigid.rotation.matrix())
            pose[:3, 3] = np.array(rigid.translation)
        # Method 2: Try direct pose attribute
        elif hasattr(img_obj, 'T'):
            pose = img_obj.T
        # Method 3: Try quaternion and translation
        elif hasattr(img_obj, 'qvec') and hasattr(img_obj, 'tvec'):
            R = self.qvec2rotmat(img_obj.qvec)
            t = img_obj.tvec
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
        
        if pose is None:
            print(f"Could not get pose for image {image.name}")
            return None
        
        # Load the image
        image_path = Path(image_dir) / image.name
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Create camera projection matrix
        P = K @ pose[:3]  # camera projection matrix

        # Step 1: Create 2D coordinate system on the painting plane
        # Create two vectors perpendicular to the normal
        if abs(plane_normal[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        v2 = np.cross(plane_normal, v1)
        v1 = np.cross(v2, plane_normal)
        
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Step 2: Project all 3D points to find the painting bounds
        points3D = []
        for point3D in recon.points3D.values():
            points3D.append(point3D.xyz)
        
        if len(points3D) < 10:
            print(f"Not enough 3D points for rectification: {image.name}")
            return None
        
        points3D = np.array(points3D)
        
        # Project all points to the 2D coordinate system
        points_2d_plane = []
        for point_3d in points3D:
            # Vector from plane center to point
            vec = point_3d - plane_center
            
            # Project to 2D coordinate system
            u = np.dot(vec, v1)
            v = np.dot(vec, v2)
            points_2d_plane.append([u, v])
        
        points_2d_plane = np.array(points_2d_plane)
        
        # Step 3: Find the rectangular envelope
        min_u, min_v = points_2d_plane.min(axis=0)
        max_u, max_v = points_2d_plane.max(axis=0)
        
        # Add margin
        margin = config.RECTIFICATION_CONFIG['envelope_margin']
        u_range = max_u - min_u
        v_range = max_v - min_v
        min_u -= u_range * margin
        max_u += u_range * margin
        min_v -= v_range * margin
        max_v += v_range * margin
        
        # Step 4: Create the rectification grid
        grid_size = config.GRID_SIZE
        rectified_points_3d = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Map grid coordinates to 2D plane coordinates
                u = min_u + (i / (grid_size - 1)) * (max_u - min_u)
                v = min_v + (j / (grid_size - 1)) * (max_v - min_v)
                
                # Convert back to 3D
                point_3d = plane_center + u * v1 + v * v2
                rectified_points_3d.append(point_3d)
        
        rectified_points_3d = np.array(rectified_points_3d)
        
        # Step 5: Project rectified points to image
        src_points = []
        dst_points = []
        
        for i, point_3d in enumerate(rectified_points_3d):
            point_homo = np.append(point_3d, 1)
            point_img = P @ point_homo
            point_img = point_img[:2] / point_img[2]
            
            # Check if point is in image bounds
            if (0 <= point_img[0] < img.shape[1] and 
                0 <= point_img[1] < img.shape[0]):
                src_points.append(point_img)
                # Map to rectified coordinates
                dst_points.append([i % grid_size, i // grid_size])
        
        if len(src_points) < 4:
            print(f"Not enough valid points for rectification: {image.name}")
            return None
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Step 6: Compute homography and apply rectification
        H = cv2.findHomography(src_points, dst_points)[0]
        rectified = cv2.warpPerspective(img, H, (grid_size, grid_size))
        
        # Step 7: Create original frame outline data
        if config.RECTIFICATION_CONFIG['show_original_frames']:
            # Project image corners to 2D plane
            img_corners = np.array([
                [0, 0],                    # Top-left
                [img.shape[1], 0],         # Top-right
                [img.shape[1], img.shape[0]], # Bottom-right
                [0, img.shape[0]]          # Bottom-left
            ], dtype=np.float32)
            
            # Project corners to 3D (approximate)
            corner_points_3d = []
            for corner in img_corners:
                # Create a ray from camera center through the corner
                ray = np.linalg.inv(K) @ np.array([corner[0], corner[1], 1])
                ray = ray / np.linalg.norm(ray)
                
                # Intersect with plane
                # Plane equation: (p - plane_center) · plane_normal = 0
                # Ray equation: p = camera_center + t * ray
                camera_center = -pose[:3, :3].T @ pose[:3, 3]
                t = np.dot(plane_center - camera_center, plane_normal) / np.dot(ray, plane_normal)
                point_3d = camera_center + t * ray
                corner_points_3d.append(point_3d)
            
            # Project to 2D coordinate system
            corner_points_2d = []
            for point_3d in corner_points_3d:
                vec = point_3d - plane_center
                u = np.dot(vec, v1)
                v = np.dot(vec, v2)
                corner_points_2d.append([u, v])
            
            # Map to rectified image coordinates
            corner_points_rectified = []
            for u, v in corner_points_2d:
                # Map to rectified image coordinates
                i = int((u - min_u) / (max_u - min_u) * (grid_size - 1))
                j = int((v - min_v) / (max_v - min_v) * (grid_size - 1))
                i = max(0, min(grid_size - 1, i))
                j = max(0, min(grid_size - 1, j))
                corner_points_rectified.append([i, j])
            
            # Store frame outline data
            frame_outline = {
                'corners_2d': corner_points_2d,
                'corners_rectified': corner_points_rectified,
                'image_name': image.name
            }
            
            # Save frame outline data
            if config.INTERMEDIATE_RESULTS['save_rectification_data']:
                self.save_intermediate_result(f"frame_outline_{image.name}", frame_outline)
        
        return rectified
    
    def compare_local_calibrations(self):
        """Compare local camera calibration results"""
        log_step("Comparison between local camera calibration results")
        
        if not self.local_camera_calibrations:
            print("No local calibrations to compare")
            return
        
        print("Local camera calibrations:")
        for painting_name, calibration in self.local_camera_calibrations.items():
            params = calibration['params']
            print(f"  {painting_name}: focal_length={params[0]:.2f}, cx={params[1]:.2f}, cy={params[2]:.2f}")
        
        # Calculate average calibration
        if len(self.local_camera_calibrations) > 1:
            avg_params = np.mean([cal['params'] for cal in self.local_camera_calibrations.values()], axis=0)
            print(f"Average calibration: focal_length={avg_params[0]:.2f}, cx={avg_params[1]:.2f}, cy={avg_params[2]:.2f}")
    
    def global_bundle_adjustment(self):
        """Global bundle adjustment (minimization of reprojection errors for all painting bundles)"""
        log_step("Global bundle adjustment")
        
        # For now, we'll use the average of local calibrations but convert to SIMPLE_RADIAL
        # In a more sophisticated implementation, you would run actual bundle adjustment
        if self.local_camera_calibrations:
            # Average the pinhole parameters
            avg_params = np.mean([cal['params'] for cal in self.local_camera_calibrations.values()], axis=0)
            ref_calibration = list(self.local_camera_calibrations.values())[0]
            
            # Convert to SIMPLE_RADIAL model for global calibration
            global_params = list(avg_params)
            if len(global_params) == 3:
                # Estimate radial distortion based on focal length
                # A rough estimate: k1 = -0.0001 * (focal_length / 1000)^2
                focal_length = global_params[0]
                estimated_k1 = -0.0001 * (focal_length / 1000) ** 2
                global_params.append(estimated_k1)
            
            self.global_camera_params = {
                'width': ref_calibration['width'],
                'height': ref_calibration['height'],
                'params': global_params,
                'model': 'SIMPLE_RADIAL'  # Use more complex model for global calibration
            }
            
            print(f"Global camera calibration updated: focal_length={global_params[0]:.2f}, k1={global_params[3]:.6f}")
        else:
            print("No local calibrations available for global adjustment")
    
    def recalculate_all_positions(self):
        """Recalculation of all camera positions using new global camera intrinsic"""
        log_step("Recalculation of all camera positions using new global camera intrinsic")
        
        if not self.global_camera_params:
            print("No global camera parameters available")
            return
        
        print("Recalculating camera positions with global calibration...")
        # This would involve re-running COLMAP with the global calibration
        # For now, we'll just log that this step would be performed
        print("(This step would re-run COLMAP with global calibration)")
    
    def remove_reflections(self, image):
        """Remove reflections using frequency domain filtering"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply bilateral filter to reduce reflections
        filtered = cv2.bilateralFilter(lab, 
                                     config.BILATERAL_FILTER_PARAMS['d'],
                                     config.BILATERAL_FILTER_PARAMS['sigma_color'],
                                     config.BILATERAL_FILTER_PARAMS['sigma_space'])
        
        # Convert back to BGR
        return cv2.cvtColor(filtered, cv2.COLOR_LAB2BGR)
    
    def create_super_resolution_fusion(self, rectified_images):
        """Create super-resolution fusion of multiple rectified images"""
        if not rectified_images:
            return None
        
        print(f"Creating super-resolution fusion from {len(rectified_images)} images...")
        
        # Align images using feature matching
        aligned_images = []
        base_image = rectified_images[0]
        
        for img in rectified_images:
            # Remove reflections
            img_clean = self.remove_reflections(img)
            
            # Simple alignment (in practice, you'd use more sophisticated methods)
            if len(aligned_images) > 0:
                # Use optical flow for alignment
                prev_gray = cv2.cvtColor(aligned_images[-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    config.OPTICAL_FLOW_PARAMS['pyr_scale'],
                    config.OPTICAL_FLOW_PARAMS['levels'],
                    config.OPTICAL_FLOW_PARAMS['winsize'],
                    config.OPTICAL_FLOW_PARAMS['iterations'],
                    config.OPTICAL_FLOW_PARAMS['poly_n'],
                    config.OPTICAL_FLOW_PARAMS['poly_sigma'],
                    config.OPTICAL_FLOW_PARAMS['flags']
                )
                
                # Apply flow
                h, w = img_clean.shape[:2]
                flow_map = np.column_stack([np.arange(w), np.arange(h)]).reshape(h, w, 2)
                flow_map = flow_map.astype(np.float32) + flow
                
                img_clean = cv2.remap(img_clean, flow_map[:, :, 0], flow_map[:, :, 1], 
                                    cv2.INTER_LINEAR)
            
            aligned_images.append(img_clean)
        
        # Create super-resolution fusion
        # Simple averaging for now (could be improved with more sophisticated methods)
        fused = np.mean(aligned_images, axis=0).astype(np.uint8)
        
        # Apply sharpening
        fused = cv2.filter2D(fused, -1, config.SHARPENING_KERNEL)
        
        return fused
    
    def extract_pose_from_image(self, img_obj):
        """Extract pose data from pycolmap Image object in serializable format"""
        try:
            # Get the camera pose
            rigid = img_obj.cam_from_world()
            
            # Extract rotation matrix and translation
            rotation_matrix = np.array(rigid.rotation.matrix())
            translation = np.array(rigid.translation)
            
            # Convert to serializable format
            return {
                'rotation_matrix': rotation_matrix.tolist(),
                'translation': translation.tolist(),
                'camera_center': (-rotation_matrix.T @ translation).tolist()
            }
        except Exception as e:
            print(f"Error extracting pose from image {img_obj.name}: {e}")
            return None

    def to_serializable(self, val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: self.to_serializable(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self.to_serializable(v) for v in val]
        # Handle pycolmap specific types
        if hasattr(val, '__class__') and 'CameraModelId' in str(val.__class__):
            return str(val)
        if hasattr(val, '__class__') and 'CameraModel' in str(val.__class__):
            return str(val)
        # Handle pycolmap Camera objects
        if hasattr(val, '__class__') and 'Camera' in str(val.__class__):
            try:
                return {
                    'width': val.width,
                    'height': val.height,
                    'params': val.params.tolist() if hasattr(val.params, 'tolist') else list(val.params),
                    'model': str(val.model)
                }
            except Exception as e:
                print(f"Error serializing Camera object: {e}")
                return str(val)
        return val

    def create_comprehensive_overview(self, results):
        """Create a comprehensive overview showing all paintings with relative positioning"""
        log_step("Creating comprehensive overview of all paintings")
        
        if not results:
            print("No results to create overview from")
            return
        
        # Load all overview images
        overview_images = {}
        painting_positions = {}
        
        for painting_name, result in results.items():
            overview_path = self.rectified_dir / f"{painting_name}_overview.jpg"
            if overview_path.exists():
                overview_img = cv2.imread(str(overview_path))
                if overview_img is not None:
                    overview_images[painting_name] = overview_img
                    # Store painting position from plane center
                    plane_center = np.array(result['plane_center'])
                    painting_positions[painting_name] = plane_center
                    print(f"Loaded overview for painting {painting_name}")
        
        if not overview_images:
            print("No overview images found")
            return
        
        # Create a large canvas for the comprehensive overview
        # Estimate canvas size based on painting positions
        if len(painting_positions) > 1:
            # Calculate relative positions
            positions = np.array(list(painting_positions.values()))
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            
            # Create a coordinate system for visualization
            canvas_width = 2000
            canvas_height = 1500
            margin = 100
            
            # Scale factor to fit all paintings
            pos_range = max_pos - min_pos
            scale_x = (canvas_width - 2 * margin) / max(pos_range[0], pos_range[1])
            scale_y = (canvas_height - 2 * margin) / max(pos_range[0], pos_range[1])
            scale = min(scale_x, scale_y)
            
            # Create canvas
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Place each painting on the canvas
            for painting_name, overview_img in overview_images.items():
                if painting_name in painting_positions:
                    pos = painting_positions[painting_name]
                    
                    # Convert 3D position to 2D canvas coordinates
                    x = int((pos[0] - min_pos[0]) * scale + margin)
                    y = int((pos[1] - min_pos[1]) * scale + margin)
                    
                    # Resize overview image to consistent size
                    target_size = 300
                    resized = cv2.resize(overview_img, (target_size, target_size))
                    
                    # Calculate position to center the image
                    x_start = max(0, x - target_size // 2)
                    y_start = max(0, y - target_size // 2)
                    x_end = min(canvas_width, x_start + target_size)
                    y_end = min(canvas_height, y_start + target_size)
                    
                    # Copy image to canvas
                    if x_end > x_start and y_end > y_start:
                        img_x_start = max(0, target_size // 2 - x)
                        img_y_start = max(0, target_size // 2 - y)
                        img_x_end = img_x_start + (x_end - x_start)
                        img_y_end = img_y_start + (y_end - y_start)
                        
                        # Ensure bounds are within the resized image
                        img_x_end = min(img_x_end, resized.shape[1])
                        img_y_end = min(img_y_end, resized.shape[0])
                        
                        # Only copy if we have valid regions and matching shapes
                        if (img_x_end > img_x_start and img_y_end > img_y_start and
                            x_end > x_start and y_end > y_start):
                            
                            # Ensure the regions have matching shapes
                            canvas_region = canvas[y_start:y_end, x_start:x_end]
                            img_region = resized[img_y_start:img_y_end, img_x_start:img_x_end]
                            
                            if canvas_region.shape == img_region.shape:
                                canvas[y_start:y_end, x_start:x_end] = img_region
                        
                        # Draw frame outline
                        cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                        
                        # Add painting label
                        cv2.putText(canvas, f"Painting {painting_name}", 
                                   (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)
            
            # Save comprehensive overview
            overview_path = self.rectified_dir / "comprehensive_overview.jpg"
            cv2.imwrite(str(overview_path), canvas)
            print(f"Comprehensive overview saved: {overview_path.name}")
            
            # Also create a simple grid layout as alternative
            self.create_grid_overview(overview_images)
        else:
            # If only one painting, just copy it
            painting_name = list(overview_images.keys())[0]
            overview_path = self.rectified_dir / "comprehensive_overview.jpg"
            cv2.imwrite(str(overview_path), overview_images[painting_name])
            print(f"Single painting overview saved: {overview_path.name}")
    
    def create_grid_overview(self, overview_images):
        """Create a grid layout overview of all paintings"""
        if not overview_images:
            return
        
        # Calculate grid dimensions
        n_paintings = len(overview_images)
        cols = int(np.ceil(np.sqrt(n_paintings)))
        rows = int(np.ceil(n_paintings / cols))
        
        # Create grid canvas
        target_size = 400
        grid_width = cols * target_size
        grid_height = rows * target_size
        grid_canvas = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Place paintings in grid
        for idx, (painting_name, overview_img) in enumerate(overview_images.items()):
            row = idx // cols
            col = idx % cols
            
            # Resize image
            resized = cv2.resize(overview_img, (target_size, target_size))
            
            # Calculate position
            x_start = col * target_size
            y_start = row * target_size
            x_end = x_start + target_size
            y_end = y_start + target_size
            
            # Copy image
            grid_canvas[y_start:y_end, x_start:x_end] = resized
            
            # Draw frame outline
            cv2.rectangle(grid_canvas, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            
            # Add painting label
            cv2.putText(grid_canvas, f"Painting {painting_name}", 
                       (x_start + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 2)
        
        # Save grid overview
        grid_path = self.rectified_dir / "grid_overview.jpg"
        cv2.imwrite(str(grid_path), grid_canvas)
        print(f"Grid overview saved: {grid_path.name}")

    def process_all_paintings(self):
        """Process all painting sets with comprehensive logging and step configuration"""
        print("Starting painting reconstruction...")
        print(f"Processing steps: {[k for k, v in config.PROCESSING_STEPS.items() if v]}")
        
        # Check for existing intermediate results
        available_results = self.check_intermediate_results()
        if available_results:
            print(f"Found existing intermediate results: {list(available_results.keys())}")
        
        # Estimate global camera calibration
        self.global_camera_params = self.estimate_global_camera_calibration()
        
        # Step 1: Run local SfM for each painting (for initial camera positions)
        local_reconstructions = {}
        
        if config.PROCESSING_STEPS['run_local_sfm']:
            for painting_set in self.painting_sets:
                painting_name = painting_set.name
                print(f"\n{'='*80}")
                print(f"Processing painting {painting_name}")
                print(f"{'='*80}")
                
                # Check if we already have local reconstruction
                existing_result = self.load_intermediate_result(f"local_reconstruction_{painting_name}")
                if existing_result and config.INTERMEDIATE_RESULTS['save_intermediate']:
                    print(f"Found existing local reconstruction for {painting_name}")
                    # For now, we'll still run SfM, but in the future we could resume from here
                
                # Run local SfM only (no rectification yet)
                recon = self.run_local_sfm_only(painting_set, painting_name)
                
                if recon is not None:
                    local_reconstructions[painting_name] = recon
                    print(f"✅ Local SfM completed for {painting_name}")
                    
                    # Save intermediate result (automatic)
                    if config.INTERMEDIATE_RESULTS['save_local_reconstructions']:
                        # Save only essential data needed for further steps
                        reconstruction_summary = {
                            'cameras': {k: self.to_serializable(v) for k, v in recon.cameras.items()},
                            'images': {k: {
                                'name': v.name,
                                'pose': self.to_serializable(self.extract_pose_from_image(v))
                            } for k, v in recon.images.items()},
                            'num_points3D': len(recon.points3D),
                            'painting_name': painting_name
                        }
                        self.save_intermediate_result(f"local_reconstruction_{painting_name}", reconstruction_summary)
                else:
                    print(f"❌ Failed to process {painting_name}")
        else:
            print("Skipping local SfM step")
        
        # Step 2: Compare local calibrations and perform global bundle adjustment
        if config.PROCESSING_STEPS['global_calibration']:
            self.compare_local_calibrations()
            self.global_bundle_adjustment()
            
            # Save intermediate result (automatic)
            if config.INTERMEDIATE_RESULTS['save_global_calibration']:
                self.save_intermediate_result("global_calibration", {
                    'global_camera_params': self.global_camera_params,
                    'local_camera_calibrations': self.local_camera_calibrations,
                    'timestamp': datetime.now().isoformat()
                })
        else:
            print("Skipping global calibration step")
        
        # Step 3: Recalculate all camera positions using global calibration
        if config.PROCESSING_STEPS['recalculate_positions']:
            self.recalculate_all_positions()
        else:
            print("Skipping position recalculation step")
        
        # Step 4: Re-run point cloud generation and rectification with global calibration
        results = {}
        if (config.PROCESSING_STEPS['point_cloud_generation'] or 
            config.PROCESSING_STEPS['rectification'] or 
            config.PROCESSING_STEPS['create_overviews']):
            
            for painting_name, local_recon in local_reconstructions.items():
                print(f"\n{'='*80}")
                print(f"Re-processing painting {painting_name} with global calibration")
                print(f"{'='*80}")
                
                result = self.process_painting_with_global_calibration(
                    local_recon, painting_name
                )
                
                if result is not None:
                    results[painting_name] = result
                    print(f"✅ Painting {painting_name} processed successfully with global calibration")
                    
                    # Save intermediate results (automatic)
                    if config.INTERMEDIATE_RESULTS['save_point_clouds']:
                        self.save_intermediate_result(f"point_cloud_{painting_name}", {
                            'plane_normal': result['plane_normal'],
                            'plane_center': result['plane_center'],
                            'painting_name': painting_name,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    if config.INTERMEDIATE_RESULTS['save_rectification_data']:
                        self.save_intermediate_result(f"rectification_data_{painting_name}", {
                            'camera_params': result['camera_params'],
                            'num_images': result['num_images'],
                            'painting_name': painting_name,
                            'timestamp': datetime.now().isoformat()
                        })
                else:
                    print(f"❌ Failed to process {painting_name} with global calibration")
        
        # Save final calibration data
        with open(self.calibration_dir / 'camera_calibration.json', 'w') as f:
            json.dump(self.to_serializable({
                'global_camera_params': self.global_camera_params,
                'local_camera_calibrations': self.local_camera_calibrations,
                'painting_results': results
            }), f, indent=2)
        
        # Step 5: Create comprehensive overview
        if config.PROCESSING_STEPS['comprehensive_overview']:
            self.create_comprehensive_overview(results)
        else:
            print("Skipping comprehensive overview step")
        
        print(f"\n{'='*80}")
        print("RECONSTRUCTION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved in: {self.output_dir}")
        print(f"Fused images: {self.fused_dir}")
        print(f"Rectified images: {self.rectified_dir}")
        print(f"Calibration data: {self.calibration_dir}")
        if config.INTERMEDIATE_RESULTS['save_intermediate']:
            print(f"Intermediate results: {self.output_dir / 'intermediate'}")
    
    def resume_from_intermediate(self, step_name):
        """Resume processing from a specific intermediate step"""
        print(f"Attempting to resume from step: {step_name}")
        
        if step_name == "local_reconstructions":
            # Load all local reconstructions
            local_reconstructions = {}
            for painting_set in self.painting_sets:
                painting_name = painting_set.name
                result = self.load_intermediate_result(f"local_reconstruction_{painting_name}")
                if result:
                    print(f"Loaded local reconstruction for {painting_name}")
                    # Note: This would need to reconstruct the pycolmap objects
                    # For now, we'll just note that we found the data
                else:
                    print(f"No local reconstruction found for {painting_name}")
        
        elif step_name == "global_calibration":
            result = self.load_intermediate_result("global_calibration")
            if result:
                self.global_camera_params = result.get('global_camera_params')
                self.local_camera_calibrations = result.get('local_camera_calibrations')
                print("Loaded global calibration data")
        
        else:
            print(f"Unknown step: {step_name}")
            return False
        
        return True
    
    def save_intermediate_result(self, name, data):
        """Save intermediate processing result"""
        if not config.INTERMEDIATE_RESULTS['save_intermediate']:
            return
        
        intermediate_dir = self.output_dir / 'intermediate'
        intermediate_dir.mkdir(exist_ok=True)
        
        # Convert data to serializable format
        serializable_data = self.to_serializable(data)
        
        # Save as JSON
        result_path = intermediate_dir / f"{name}.json"
        with open(result_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Saved intermediate result: {result_path.name}")
    
    def load_intermediate_result(self, name):
        """Load intermediate processing result"""
        intermediate_dir = self.output_dir / 'intermediate'
        result_path = intermediate_dir / f"{name}.json"
        
        if result_path.exists():
            try:
                with open(result_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading intermediate result {name}: {e}")
                print(f"File may be corrupted: {result_path}")
                return None
        else:
            print(f"Intermediate result not found: {result_path}")
            return None
    
    def check_intermediate_results(self):
        """Check which intermediate results are available"""
        intermediate_dir = self.output_dir / 'intermediate'
        if not intermediate_dir.exists():
            return {}
        
        available_results = {}
        for result_file in intermediate_dir.glob("*.json"):
            result_name = result_file.stem
            available_results[result_name] = str(result_file)
        
        return available_results
    
    def add_frame_outlines_to_overview(self, overview, painting_name):
        """Add original frame outlines to the overview image"""
        # Load frame outline data for this painting
        intermediate_dir = self.output_dir / 'intermediate'
        frame_outlines = []
        
        # Find all frame outline files for this painting
        for outline_file in intermediate_dir.glob(f"frame_outline_*_{painting_name}_*.json"):
            try:
                with open(outline_file, 'r') as f:
                    frame_outline = json.load(f)
                    frame_outlines.append(frame_outline)
            except:
                continue
        
        # Draw frame outlines on overview
        for frame_outline in frame_outlines:
            corners = frame_outline.get('corners_rectified', [])
            if len(corners) >= 4:
                # Convert to integer coordinates
                corners_int = np.array(corners, dtype=np.int32)
                
                # Draw the frame outline
                cv2.polylines(overview, [corners_int], True, (0, 255, 0), 2)
                
                # Add image name label
                image_name = frame_outline.get('image_name', 'unknown')
                if corners_int.shape[0] > 0:
                    label_pos = tuple(corners_int[0])
                    cv2.putText(overview, image_name, label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return overview

if __name__ == "__main__":
    reconstructor = PaintingReconstructor()
    reconstructor.process_all_paintings()