import cv2
import numpy as np
import os
import glob
import json
from datetime import datetime

# ===== CONFIGURATION SECTION =====
CHECKERBOARD = (21, 13)  # inner corners. Ex: 7 by 10 squares = (6,9)
SQUARE_SIZE = 2.0  # inches
RAW_IMAGES_DIR = './raw_images'
BASE_OUTPUT_DIR = './annotated_images'
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# ===== END CONFIGURATION =====

# Validate raw images directory exists
if not os.path.exists(RAW_IMAGES_DIR):
    print(f"Error: Images directory '{RAW_IMAGES_DIR}' not found!")
    exit(1)

# Get all images
images = glob.glob(os.path.join(RAW_IMAGES_DIR, '*.jpg'))
if len(images) == 0:
    print(f"Error: No .jpg images found in '{RAW_IMAGES_DIR}'!")
    exit(1)

print(f"Found {len(images)} images to process.")

# Create timestamped output directory
timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
output_dir = os.path.join(BASE_OUTPUT_DIR, timestamp)
os.makedirs(output_dir, exist_ok=True)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
prev_img_shape = None

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Validate all images have same resolution
print("Validating image resolutions...")
valid_images = []
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Failed to load {os.path.basename(fname)}")
        continue
    if prev_img_shape is None:
        prev_img_shape = img.shape
    elif img.shape != prev_img_shape:
        print(f"Warning: {os.path.basename(fname)} has different resolution {img.shape} vs {prev_img_shape}")
        continue
    valid_images.append(fname)

images = valid_images
if len(images) == 0:
    print("Error: No valid images with matching resolution!")
    exit(1)

print(f"Using {len(images)} images with uniform resolution.")

annotated_count = 0
failed_count = 0
last_gray = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    last_gray = gray
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # Save annotated image
        basename = os.path.basename(fname)
        save_path = os.path.join(output_dir, f"annotated_{basename}")
        cv2.imwrite(save_path, img)
        print(f"  ✓ Saved: {os.path.basename(save_path)}")
        annotated_count += 1
    else:
        basename = os.path.basename(fname)
        print(f"  ✗ Checkerboard not detected: {basename}")
        failed_count += 1

print(f"\n--- Image Processing Summary ---")
print(f"Raw images processed: {len(images)}")
print(f"Annotated images saved: {annotated_count}")
print(f"Failed to detect checkerboard: {failed_count}")

if len(objpoints) == 0:
    print("Error: No valid checkerboards found. Cannot calibrate camera.")
    exit(1)

if last_gray is None:
    print("Error: No valid images processed.")
    exit(1)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, last_gray.shape[::-1], None, None)

# Calculate reprojection error
reprojection_errors = []
total_error = 0

for i in range(len(objpoints)):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    reprojection_errors.append(float(error))
    total_error += error

mean_reprojection_error = total_error / len(objpoints)

# Prepare error data
error_data = {
    "Calibration Date/Time": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
    "Mean Reprojection Error (pixels)": float(mean_reprojection_error),
    "Per-Image Reprojection Errors (pixels)": reprojection_errors,
    "Number of Images Used": annotated_count,
    "Image Resolution": [last_gray.shape[1], last_gray.shape[0]],
    "Checkerboard Dimensions": CHECKERBOARD,
    "Square Size (inches)": SQUARE_SIZE
}

# Save error data to json file
error_file = "error.json"
with open(error_file, 'w') as f:
    json.dump(error_data, f, indent=4)
print(f"\nError analysis saved to: {error_file}")
print(f"Mean reprojection error: {mean_reprojection_error:.4f} pixels")
if mean_reprojection_error < 0.5:
    print("✓ Excellent calibration quality!")
elif mean_reprojection_error < 1.0:
    print("✓ Good calibration quality")
else:
    print("⚠ Warning: High reprojection error, calibration may be suboptimal")

# Prepare calibration data
calibration_data = {
    "Calibration Date/Time": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
    "Camera Matrix (mtx or K)": mtx.tolist(),
    "Distortion Coefficients (dist)": dist.tolist(),
    "Rotation Vectors (rvecs)": [rv.tolist() for rv in rvecs],
    "Translation Vectors (tvecs)": [tv.tolist() for tv in tvecs],
    "Calibration Metadata": {
        "Checkerboard Dimensions": CHECKERBOARD,
        "Square Size (inches)": SQUARE_SIZE,
        "Number of Images": annotated_count,
        "Image Resolution": [last_gray.shape[1], last_gray.shape[0]]
    }
}

# Save to json file
calibration_file = "results.json"
with open(calibration_file, 'w') as f:
    json.dump(calibration_data, f, indent=4)
print(f"Calibration results saved to: {calibration_file}")
print(f"Annotated images saved to: {output_dir}")
print("\n✓ Calibration complete!")