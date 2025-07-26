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
warnings.filterwarnings('ignore')
import config

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
        
    def run_colmap_sfm(self, image_dir, output_path, database_path):
        """Run COLMAP SfM on a set of images"""
        try:
            # Create database
            database = pycolmap.Database(str(database_path))
            
            # Import images to database
            pycolmap.import_images(str(database_path), str(image_dir))
            
            # Extract features
            pycolmap.extract_features(str(database_path), str(image_dir))
            
            # Match features
            pycolmap.match_exhaustive(str(database_path))
            
            # Run incremental mapping
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(image_dir),
                output_path=str(output_path)
            )
            
            # Return the first (and usually only) reconstruction
            if reconstructions:
                return list(reconstructions.values())[0]
            return None
        except Exception as e:
            print(f"COLMAP failed for {image_dir}: {e}")
            return None
    
    def estimate_global_camera_calibration(self):
        """Estimate global camera calibration from all painting sets"""
        print("Estimating global camera calibration...")
        
        all_cameras = []
        for painting_set in self.painting_sets:
            temp_output = self.output_dir / f"temp_{painting_set.name}"
            temp_output.mkdir(exist_ok=True)
            
            recon = self.run_colmap_sfm(
                painting_set, 
                temp_output, 
                temp_output / 'sfm.db'
            )
            
            if recon and recon.cameras:
                # Get camera parameters from this set
                for camera_id, camera in recon.cameras.items():
                    all_cameras.append({
                        'width': camera.width,
                        'height': camera.height,
                        'params': camera.params.tolist() if hasattr(camera.params, 'tolist') else list(camera.params),
                        'model': str(camera.model)
                    })
        
        if not all_cameras:
            print("No camera data found. Using default parameters.")
            return config.DEFAULT_CAMERA_PARAMS
        
        # Use the most common camera parameters
        # For simplicity, we'll use the first camera's parameters
        # In a more sophisticated approach, you could average parameters
        return all_cameras[0]
    
    def process_painting_set(self, painting_set_path, painting_name):
        """Process a single painting set"""
        print(f"\nProcessing painting set: {painting_name}")
        
        # Run COLMAP SfM
        recon = self.run_colmap_sfm(
            painting_set_path, 
            self.output_dir / painting_name,
            self.output_dir / painting_name / 'database.db'
        )
        
        if not recon:
            print(f"❌ COLMAP reconstruction failed for {painting_name}")
            return None
        
        # Find the painting plane
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
        rectified_images = []
        image_files = list(painting_set_path.glob('*.jpg')) + list(painting_set_path.glob('*.jpeg')) + list(painting_set_path.glob('*.png'))
        
        for image_file in image_files:
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
        
        if not rectified_images:
            print(f"❌ No images could be rectified for {painting_name}")
            return None
        
        # Create fusion
        fused = self.create_fusion(rectified_images)
        if fused is not None:
            # Save fused image
            output_path = self.fused_dir / f"{painting_name}_fused.jpg"
            cv2.imwrite(str(output_path), fused)
            print(f"✅ Created fusion for {painting_name}")
            return {
                'painting_name': painting_name,
                'num_images': len(rectified_images),
                'plane_normal': plane_normal.tolist(),
                'plane_center': plane_center.tolist(),
                'camera_params': self.to_serializable(camera_params)
            }
        else:
            print(f"❌ Failed to create fusion for {painting_name}")
            return None
    
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
        """Rectify image to painting plane"""
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
        """Process all painting sets"""
        print("Starting painting reconstruction...")
        
        # Estimate global camera calibration
        self.global_camera_params = self.estimate_global_camera_calibration()
        
        results = {}
        
        for painting_set in self.painting_sets:
            painting_name = painting_set.name
            print(f"\n{'='*50}")
            print(f"Processing painting {painting_name}")
            print(f"{'='*50}")
            
            # Process this painting set
            result = self.process_painting_set(painting_set, painting_name)
            
            if result is not None:
                # The result structure is now different, so we need to adjust how we access data
                painting_name = result['painting_name']
                num_images = result['num_images']
                plane_normal = np.array(result['plane_normal'])
                plane_center = np.array(result['plane_center'])
                camera_params = result['camera_params']
                
                # Create super-resolution fusion
                fused_image = self.create_super_resolution_fusion(None) # Pass None as rectified_images are saved
                
                if fused_image is not None:
                    # Save results
                    cv2.imwrite(str(self.fused_dir / f"{painting_name}_fused.jpg"), fused_image)
                    
                    # Save individual rectified images
                    # The rectified images are already saved in process_painting_set
                    
                    results[painting_name] = {
                        'camera_matrix': camera_params['params'], # Assuming params is the camera matrix
                        'plane_normal': plane_normal.tolist(),
                        'plane_center': plane_center.tolist(),
                        'num_images': num_images
                    }
                    
                    print(f"✅ Painting {painting_name} processed successfully")
                    print(f"   - {num_images} images rectified")
                    print(f"   - Super-resolution fusion created")
                else:
                    print(f"❌ Failed to create fusion for {painting_name}")
            else:
                print(f"❌ Failed to process {painting_name}")
        
        # Save calibration data
        with open(self.calibration_dir / 'camera_calibration.json', 'w') as f:
            json.dump(self.to_serializable({
                'global_camera_params': self.global_camera_params,
                'painting_results': results
            }), f, indent=2)
        
        print(f"\n{'='*50}")
        print("RECONSTRUCTION COMPLETE")
        print(f"{'='*50}")
        print(f"Results saved in: {self.output_dir}")
        print(f"Fused images: {self.fused_dir}")
        print(f"Rectified images: {self.rectified_dir}")
        print(f"Calibration data: {self.calibration_dir}")

if __name__ == "__main__":
    reconstructor = PaintingReconstructor()
    reconstructor.process_all_paintings()
