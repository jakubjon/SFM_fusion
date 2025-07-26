## 1. Global Initialization

1.1 Determine preliminary global camera intrinsics:
  - Extract from EXIF of the first image if available  
  - Otherwise, default to zeros or neutral initial values

---

## 2. Registration of Painting X

2.1 Feature extraction from all images in Painting X  
  - *(Skip if already exists)*

2.2 Feature matching between all image pairs in Painting X  
  - *(Skip if already exists)*

2.3 Estimate preliminary camera positions for a single bundle in Painting X  
  - Compute reprojection error

2.4 Perform local bundle adjustment and estimate local camera intrinsics

2.5 Recalculate all camera positions using the refined local camera intrinsics

---

## 3. Registration of Painting Y

3.1 Repeat steps 2.1 through 2.5 for Painting Y

---

## 4. Cross-Bundle Calibration and Adjustment

4.1 Compare local camera calibration results across all paintings

4.2 Perform global bundle adjustment using all bundles  
  - Minimize overall reprojection error  
  - Output globally adjusted camera intrinsics

---

## 5. Final Camera Pose Recalculation

5.1 Recalculate all camera positions using the global camera intrinsics

---

## 6. Processing of Painting X

6.1 Generate point cloud

6.2 Determine painting plane

6.3 Create a reduced orthorectified painting overview  
  - Simulate infinite focal length and perpendicular viewing rays  
  - Save overview to disk

---

<!-- 
## 7. Further Processing (Skipped for Now)

7.1 Surface reconstruction

7.2 Texture mapping

7.3 Reflection removal

7.4 Super-resolution fusion

7.5 Image stitching and color correction
-->
