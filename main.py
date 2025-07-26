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
                return exif_calibration
        
        print("No EXIF calibration found, using default parameters")
        return config.DEFAULT_CAMERA_PARAMS
    
    def process_painting_set(self, painting_set_path, painting_name):
        """Process a single painting set with detailed logging"""
        log_step("Processing painting set", painting_name)
        
        # Run COLMAP SfM
        recon = self.run_colmap_sfm(
            painting_set_path, 
            self.output_dir / painting_name,
            self.output_dir / painting_name / 'database.db',
            painting_name
        )
        
        if not recon:
            print(f"❌ COLMAP reconstruction failed for {painting_name}")
            return None
        
        # Store local camera calibration
        if recon.cameras:
            ref_camera_id = list(recon.cameras.keys())[0]
            camera = recon.cameras[ref_camera_id]
            self.local_camera_calibrations[painting_name] = {
                'width': camera.width,
                'height': camera.height,
                'params': camera.params.tolist() if hasattr(camera.params, 'tolist') else list(camera.params),
                'model': str(camera.model)
            }
            print(f"Local camera calibration stored for {painting_name}")
        
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
        """Rectify image to painting plane (orthorectification)"""
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

        # Create rectification grid
        grid_size = config.GRID_SIZE
        
        # Define the rectification plane
        # We'll create a grid on the plane
        plane_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Create a point on the plane
                u = (i - grid_size/2) * 0.01  # 1cm spacing
                v = (j - grid_size/2) * 0.01
                
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
                
                # Point on plane
                point = plane_center + u * v1 + v * v2
                plane_points.append(point)
        
        plane_points = np.array(plane_points)
        
        # Project points to image
        points_2d = []
        valid_points = []
        
        for point_3d in plane_points:
            # Homogeneous coordinates
            point_homo = np.append(point_3d, 1)
            
            # Project to image
            point_img = P @ point_homo
            point_img = point_img[:2] / point_img[2]
            
            # Check if point is in image bounds
            if (0 <= point_img[0] < img.shape[1] and 
                0 <= point_img[1] < img.shape[0]):
                points_2d.append(point_img)
                valid_points.append(point_3d)
        
        if len(points_2d) < 4:
            print(f"Not enough valid points for rectification: {image.name}")
            return None
        
        points_2d = np.array(points_2d)
        valid_points = np.array(valid_points)
        
        # Create rectified image using perspective transform
        # Find the bounding box of the projected points
        min_x, min_y = points_2d.min(axis=0)
        max_x, max_y = points_2d.max(axis=0)
        
        # Create destination points for rectification
        dst_points = []
        src_points = []
        
        for i in range(0, len(points_2d), max(1, len(points_2d)//100)):  # Sample points
            src_points.append(points_2d[i])
            
            # Map to rectified coordinates
            u = (points_2d[i][0] - min_x) / (max_x - min_x) * grid_size
            v = (points_2d[i][1] - min_y) / (max_y - min_y) * grid_size
            dst_points.append([u, v])
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography
        H = cv2.findHomography(src_points, dst_points)[0]
        
        # Apply rectification
        rectified = cv2.warpPerspective(img, H, (grid_size, grid_size))
        
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
        
        # For now, we'll use the average of local calibrations
        # In a more sophisticated implementation, you would run actual bundle adjustment
        if self.local_camera_calibrations:
            avg_params = np.mean([cal['params'] for cal in self.local_camera_calibrations.values()], axis=0)
            ref_calibration = list(self.local_camera_calibrations.values())[0]
            
            self.global_camera_params = {
                'width': ref_calibration['width'],
                'height': ref_calibration['height'],
                'params': avg_params.tolist(),
                'model': ref_calibration['model']
            }
            
            print(f"Global camera calibration updated: focal_length={avg_params[0]:.2f}")
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
        return val

    def process_all_paintings(self):
        """Process all painting sets with comprehensive logging"""
        print("Starting painting reconstruction...")
        
        # Estimate global camera calibration
        self.global_camera_params = self.estimate_global_camera_calibration()
        
        results = {}
        
        # Process each painting set
        for painting_set in self.painting_sets:
            painting_name = painting_set.name
            print(f"\n{'='*80}")
            print(f"Processing painting {painting_name}")
            print(f"{'='*80}")
            
            # Process this painting set
            result = self.process_painting_set(painting_set, painting_name)
            
            if result is not None:
                results[painting_name] = result
                print(f"✅ Painting {painting_name} processed successfully")
            else:
                print(f"❌ Failed to process {painting_name}")
        
        # Compare local calibrations
        self.compare_local_calibrations()
        
        # Global bundle adjustment
        self.global_bundle_adjustment()
        
        # Recalculate all positions
        self.recalculate_all_positions()
        
        # Save calibration data
        with open(self.calibration_dir / 'camera_calibration.json', 'w') as f:
            json.dump(self.to_serializable({
                'global_camera_params': self.global_camera_params,
                'local_camera_calibrations': self.local_camera_calibrations,
                'painting_results': results
            }), f, indent=2)
        
        print(f"\n{'='*80}")
        print("RECONSTRUCTION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved in: {self.output_dir}")
        print(f"Fused images: {self.fused_dir}")
        print(f"Rectified images: {self.rectified_dir}")
        print(f"Calibration data: {self.calibration_dir}")

if __name__ == "__main__":
    reconstructor = PaintingReconstructor()
    reconstructor.process_all_paintings()
