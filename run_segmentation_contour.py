
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO

def draw_points(image: np.ndarray, points: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    output = image.copy()
    for key in keys:
        if key in points:
            pt = points[key]
            cv2.circle(output, tuple(pt.astype(int)), 6, (255, 0, 0), -1)
    return output

# --- Insert missing detect_mirror_points function ---
def detect_mirror_points(
    gray: np.ndarray, contour: np.ndarray
) -> dict[str, np.ndarray]:
    height, width = gray.shape[:2]
    x_min = int(np.min(contour[:, 0]))
    x_max = int(np.max(contour[:, 0]))
    y_min = int(np.min(contour[:, 1]))
    y_max = int(np.max(contour[:, 1]))
    door_width = max(1, x_max - x_min)
    door_height = max(1, y_max - y_min)

    # Search the left side of the door where the full mirror assembly lives.
    roi_x0 = max(0, x_min - int(0.10 * door_width))
    roi_x1 = min(width, x_min + int(0.30 * door_width))
    roi_y0 = max(0, y_min + int(0.12 * door_height))
    roi_y1 = min(height, y_min + int(0.68 * door_height))
    roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
    if roi.size == 0:
        return {}

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    expected = np.array([x_min + 0.09 * door_width, y_min + 0.30 * door_height], dtype=np.float32)
    best_contour = None
    best_score = None
    for candidate in contours:
        area = cv2.contourArea(candidate)
        if area < 2500:
            continue
        x, y, w, h = cv2.boundingRect(candidate)
        global_x = x + roi_x0
        global_y = y + roi_y0
        if h < 120 or w < 50:
            continue
        if global_x > x_min + 0.20 * door_width:
            continue
        if global_y < y_min + 0.10 * door_height or global_y > y_min + 0.48 * door_height:
            continue
        center = np.array([global_x + w / 2.0, global_y + h / 2.0], dtype=np.float32)
        distance = float(np.linalg.norm(center - expected))
        score = area + 4.0 * h - 18.0 * distance - 2.0 * (global_x - x_min)
        if best_score is None or score > best_score:
            best_score = score
            best_contour = candidate

    if best_contour is None:
        return {}

    bx, by, bw, bh = cv2.boundingRect(best_contour)
    bbox_top = by + roi_y0
    bbox_left = bx + roi_x0

    assembly_global = best_contour + np.array([[[roi_x0, roi_y0]]], dtype=np.int32)
    contour_points = assembly_global.reshape(-1, 2)
    perimeter = cv2.arcLength(assembly_global, True)
    approx = cv2.approxPolyDP(assembly_global, 0.012 * perimeter, True).reshape(-1, 2)
    lower_y = bbox_top + 0.72 * bh

    best_segment = None
    best_score = None
    for index in range(len(approx)):
        start = approx[index].astype(np.float32)
        end = approx[(index + 1) % len(approx)].astype(np.float32)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < max(20.0, 0.18 * bw):
            continue
        if min(start[1], end[1]) < lower_y:
            continue
        horizontal_ratio = abs(float(delta[0])) / max(length, 1e-6)
        vertical_delta = abs(float(delta[1]))
        if horizontal_ratio < 0.93 or vertical_delta > max(14.0, 0.06 * bh):
            continue
        mid_x = float((start[0] + end[0]) / 2.0)
        mid_y = float((start[1] + end[1]) / 2.0)
        score = 1.8 * length + 2.2 * (mid_y - bbox_top) - 0.10 * abs(mid_x - (bbox_left + 0.5 * bw))
        if best_score is None or score > best_score:
            best_score = score
            best_segment = (start, end)

    if best_segment is None:
        lower_points = contour_points[contour_points[:, 1] >= lower_y]
        if len(lower_points) < 2:
            return {}
        y_target = int(np.percentile(lower_points[:, 1], 82))
        band = lower_points[np.abs(lower_points[:, 1] - y_target) <= max(8, int(0.04 * bh))]
        if len(band) < 2:
            band = lower_points
        left_point = band[np.argmin(band[:, 0])].astype(np.int32)
        right_point = band[np.argmax(band[:, 0])].astype(np.int32)
    else:
        start, end = best_segment
        if start[0] <= end[0]:
            left_point = np.rint(start).astype(np.int32)
            right_point = np.rint(end).astype(np.int32)
        else:
            left_point = np.rint(end).astype(np.int32)
            right_point = np.rint(start).astype(np.int32)

    return {
        "mirror_mount_left": left_point.astype(np.int32),
        "mirror_mount_right": right_point.astype(np.int32),
        "mirror_mount_corner": right_point.astype(np.int32),
    }


def intersect_lines(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    a1, b1, c1 = line_coeffs_from_points(first[0], first[1])
    a2, b2, c2 = line_coeffs_from_points(second[0], second[1])
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        raise RuntimeError("Refined lines are nearly parallel and cannot be intersected")
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)

def project_point_to_line(point: np.ndarray, line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    p1 = line[0].astype(np.float32)
    p2 = line[1].astype(np.float32)
    direction = p2 - p1
    denom = float(np.dot(direction, direction))
    if denom == 0:
        raise RuntimeError("Cannot project onto a zero-length line")
    scale = float(np.dot(point.astype(np.float32) - p1, direction) / denom)
    return p1 + scale * direction


def get_cad_points_3d() -> np.ndarray:
    # CAD coordinates, origin at bottom_left (0,0,0)
    # These values are from your notebook, adjust as needed
    bottom_right = np.array([41.255, 0.0])
    top_right    = np.array([42.222, 62.269])
    right_25 = bottom_right + 0.25 * (top_right - bottom_right)
    right_50 = bottom_right + 0.50 * (top_right - bottom_right)
    right_75 = bottom_right + 0.75 * (top_right - bottom_right)
    points_3d = np.array([
        [0.000, 0.000, 0],        # bottom_left
        [10.314, 0.000, 0],       # bottom_25
        [20.628, 0.000, 0],       # bottom_50
        [30.941, 0.000, 0],       # bottom_75
        [41.255, 0.000, 0],       # bottom_right
        [right_25[0], right_25[1], 0],
        [right_50[0], right_50[1], 0],
        [right_75[0], right_75[1], 0],
        [42.222, 62.269, 0],      # top_right
    ], dtype=np.float32)
    return points_3d

def get_camera_matrix_and_dist(img_shape) -> tuple[np.ndarray, np.ndarray]:
    # Approximate camera matrix and distortion coefficients
    # These are placeholders, to be replaced with calibration results
    K = np.array([
        [1200, 0, 524],
        [0, 1200, 932],
        [0, 0, 1]
    ], dtype=np.float32)
    dist = np.array([-0.12, 0.03, 0.001, 0.0005, 0], dtype=np.float32)
    return K, dist

def get_points_2d_from_dict(points: dict[str, np.ndarray]) -> np.ndarray:
    # Order: bottom_left, bottom_25, bottom_50, bottom_75, bottom_right, right_25, right_50, right_75, top_right
    return np.array([
        points["bottom_left"],
        points["bottom_25"],
        points["bottom_50"],
        points["bottom_75"],
        points["bottom_right"],
        points["right_25"],
        points["right_50"],
        points["right_75"],
        points["top_right"],
    ], dtype=np.float32)

def draw_detected_and_projected(img: np.ndarray, detected: np.ndarray, projected: np.ndarray, detected_color=(255,0,0), projected_color=(0,0,255)) -> np.ndarray:
    out = img.copy()
    for pt in detected:
        cv2.circle(out, tuple(np.round(pt).astype(int)), 6, detected_color, -1)
    for pt in projected:
        cv2.circle(out, tuple(np.round(pt).astype(int)), 6, projected_color, -1)
    for p_true, p_proj in zip(detected, projected):
        p1 = tuple(np.round(p_true).astype(int))
        p2 = tuple(np.round(p_proj).astype(int))
        cv2.line(out, p1, p2, (0,255,0), 2)
    return out

def print_reprojection_errors(labels, detected, projected, label):
    errors = np.linalg.norm(detected - projected, axis=1)
    return errors, np.mean(errors)



def extract_points(contour: np.ndarray) -> dict[str, np.ndarray]:
    bottom_y = np.percentile(contour[:, 1], 99)
    bottom_band = contour[contour[:, 1] > bottom_y - 40]
    if len(bottom_band) < 2:
        raise ValueError("Not enough contour points in bottom band")

    bottom_left = bottom_band[np.argmin(bottom_band[:, 0])]
    bottom_right = bottom_band[np.argmax(bottom_band[:, 0])]

    bottom_50 = ((bottom_left + bottom_right) / 2).astype(np.int32)
    bottom_25 = (bottom_left + 0.25 * (bottom_right - bottom_left)).astype(np.int32)
    bottom_75 = (bottom_left + 0.75 * (bottom_right - bottom_left)).astype(np.int32)

    top_y = np.min(contour[:, 1])
    top_pts = contour[contour[:, 1] < top_y + 20]
    if len(top_pts) == 0:
        raise ValueError("Not enough contour points in top band")
    top_right = top_pts[np.argmax(top_pts[:, 0])]

    right_50 = ((bottom_right + top_right) / 2).astype(np.int32)
    right_25 = (bottom_right + 0.25 * (top_right - bottom_right)).astype(np.int32)
    right_75 = (bottom_right + 0.75 * (top_right - bottom_right)).astype(np.int32)

    return {
        "bottom_left": bottom_left.astype(np.int32),
        "bottom_right": bottom_right.astype(np.int32),
        "bottom_50": bottom_50,
        "bottom_25": bottom_25,
        "bottom_75": bottom_75,
        "top_right": top_right.astype(np.int32),
        "right_50": right_50,
        "right_25": right_25,
        "right_75": right_75,
    }


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), mask_uint8)


def line_coeffs_from_points(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    x1, y1 = p1.astype(np.float64)
    x2, y2 = p2.astype(np.float64)
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def fit_line_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise RuntimeError("Need at least two points to fit a line")
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([vx, vy], dtype=np.float32)
    origin = np.array([x0, y0], dtype=np.float32)
    return origin - 1000.0 * direction, origin + 1000.0 * direction


def sample_inside_boundary_points(
    gray: np.ndarray,
    mask_inside: np.ndarray,
    boundary_mask: np.ndarray,
    roi: tuple[int, int, int, int],
    orientation: str,
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    inside_patch = mask_inside[y0:y1, x0:x1]
    boundary_patch = boundary_mask[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError(f"Empty ROI for {orientation} point sampling")

    edges = cv2.Canny(cv2.GaussianBlur(patch, (5, 5), 0), 50, 150)
    candidate = cv2.bitwise_and(edges, boundary_patch)
    candidate = cv2.dilate(candidate, np.ones((3, 3), np.uint8), iterations=1)

    points: list[list[float]] = []
    if orientation == "vertical":
        for row in range(candidate.shape[0]):
            cols = np.where(candidate[row] > 0)[0]
            if cols.size == 0:
                continue
            # Prefer the first boundary-adjacent edge that still lies inside the door mask.
            for col in np.sort(cols):
                left = max(0, col - 3)
                right = min(inside_patch.shape[1], col + 2)
                if np.any(inside_patch[row, left:right] > 0):
                    points.append([x0 + col, y0 + row])
                    break
    elif orientation == "horizontal":
        for col in range(candidate.shape[1]):
            rows = np.where(candidate[:, col] > 0)[0]
            if rows.size == 0:
                continue
            # Prefer the topmost edge near the bottom boundary that still lies inside the mask.
            for row in np.sort(rows):
                top = max(0, row - 3)
                bottom = min(inside_patch.shape[0], row + 2)
                if np.any(inside_patch[top:bottom, col] > 0):
                    points.append([x0 + col, y0 + row])
                    break
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    if len(points) < 8:
        raise RuntimeError(f"Not enough sampled points for {orientation} fit")
    return np.array(points, dtype=np.float32)

def detect_dominant_line(
    gray: np.ndarray,
    boundary_mask: np.ndarray,
    mask_inside: np.ndarray,
    roi: tuple[int, int, int, int],
    orientation: str,
    expected_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def score_candidates(
        candidate_lines: np.ndarray | None,
        require_inside: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        best_score_local = None
        best_points_local = None
        if candidate_lines is None:
            return None

        for entry in candidate_lines[:, 0]:
            x_start, y_start, x_end, y_end = entry.astype(np.float32)
            p1 = np.array([x_start + x0, y_start + y0], dtype=np.float32)
            p2 = np.array([x_end + x0, y_end + y0], dtype=np.float32)
            delta = p2 - p1
            length = float(np.linalg.norm(delta))
            if length < 20:
                continue

            dx = abs(float(delta[0]))
            dy = abs(float(delta[1]))
            if orientation == "horizontal" and dx < dy * 2.0:
                continue
            if orientation == "vertical" and dy < dx * 1.3:
                continue

            midpoint = (p1 + p2) / 2.0
            distance = float(np.linalg.norm(midpoint - expected))
            score = length - 1.5 * distance
            if orientation == "vertical":
                samples = np.linspace(0.0, 1.0, 25, dtype=np.float32)
                inside_count = 0
                x_bias_total = 0.0
                valid_samples = 0
                for t in samples:
                    sample = p1 + t * delta
                    sx = int(round(sample[0])) - x0
                    sy = int(round(sample[1])) - y0
                    if 0 <= sx < inside_patch.shape[1] and 0 <= sy < inside_patch.shape[0]:
                        valid_samples += 1
                        if inside_patch[sy, sx] > 0:
                            inside_count += 1
                            x_bias_total += sx
                if valid_samples == 0:
                    continue
                inside_ratio = inside_count / valid_samples
                if require_inside and inside_count == 0:
                    continue
                mean_inside_x = x_bias_total / max(inside_count, 1)
                score += 80.0 * inside_ratio
                score -= 0.35 * mean_inside_x
            if best_score_local is None or score > best_score_local:
                best_score_local = score
                best_points_local = (p1, p2)
        return best_points_local

    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    boundary_patch = boundary_mask[y0:y1, x0:x1]
    inside_patch = mask_inside[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError(f"Empty ROI for {orientation} line detection")

    blur = cv2.GaussianBlur(patch, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_and(edges, boundary_patch)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=max(25, min(patch.shape[:2]) // 3),
        maxLineGap=20,
    )
    if lines is None:
        raise RuntimeError(f"No {orientation} line candidates found")

    expected = expected_point.astype(np.float32)
    best_points = score_candidates(lines, require_inside=(orientation == "vertical"))
    if best_points is None and orientation == "vertical":
        best_points = score_candidates(lines, require_inside=False)
    if best_points is None:
        raise RuntimeError(f"No usable {orientation} line candidates survived filtering")
    return best_points

def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation and contour extraction on one image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="v6.pt", help="Path to the YOLO model weights")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated files")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), verbose=False)
    if not results:
        raise RuntimeError("Model returned no results")

    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise RuntimeError("No segmentation mask detected in the image")

    mask = result.masks.data[0].cpu().numpy()
    poly = result.masks.xy[0].astype(np.int32)
    original = result.orig_img.copy()

    epsilon = 0.0015 * cv2.arcLength(poly, True)
    smooth = cv2.approxPolyDP(poly, epsilon, True)
    contour = smooth.reshape(-1, 2)
    points = extract_points(contour)

    stem = image_path.stem
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_uint8 = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
    boundary_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    mask_inside = cv2.erode(mask_uint8, np.ones((9, 9), np.uint8), iterations=1)
    bottom_margin = 45
    right_margin = 45

    bottom_roi = (
        max(0, int(points["bottom_left"][0]) - 20),
        max(0, int(min(points["bottom_left"][1], points["bottom_right"][1])) - bottom_margin),
        min(width, int(points["bottom_right"][0]) + 20),
        min(height, int(max(points["bottom_left"][1], points["bottom_right"][1])) + bottom_margin),
    )
    right_roi = (
        max(0, int(min(points["bottom_right"][0], points["top_right"][0])) - right_margin),
        max(0, int(points["top_right"][1]) - 20),
        min(width, int(max(points["bottom_right"][0], points["top_right"][0])) + right_margin),
        min(height, int(points["bottom_right"][1]) + 20),
    )

    try:
        bottom_samples = sample_inside_boundary_points(
            gray,
            mask_inside,
            boundary_mask,
            bottom_roi,
            "horizontal",
        )
        refined_bottom = fit_line_from_points(bottom_samples)
    except RuntimeError:
        refined_bottom = detect_dominant_line(
            gray,
            boundary_mask,
            mask_inside,
            bottom_roi,
            "horizontal",
            expected_point=(points["bottom_left"] + points["bottom_right"]) / 2.0,
        )

    try:
        right_samples = sample_inside_boundary_points(
            gray,
            mask_inside,
            boundary_mask,
            right_roi,
            "vertical",
        )
        refined_right = fit_line_from_points(right_samples)
    except RuntimeError:
        refined_right = detect_dominant_line(
            gray,
            boundary_mask,
            mask_inside,
            right_roi,
            "vertical",
            expected_point=(points["bottom_right"] + points["top_right"]) / 2.0,
        )

    refined_bottom_right = intersect_lines(refined_bottom, refined_right)
    refined_bottom_left = project_point_to_line(points["bottom_left"], refined_bottom)
    refined_top_right = project_point_to_line(points["top_right"], refined_right)

    points["bottom_left"] = np.rint(refined_bottom_left).astype(np.int32)
    points["bottom_right"] = np.rint(refined_bottom_right).astype(np.int32)
    points["top_right"] = np.rint(refined_top_right).astype(np.int32)
    points["bottom_50"] = np.rint((refined_bottom_left + refined_bottom_right) / 2.0).astype(np.int32)
    mirror_points = detect_mirror_points(gray, contour)
    points.update(mirror_points)

    # --- Homography and solvePnP projection/visualization ---
    points_2d = get_points_2d_from_dict(points)
    points_3d = get_cad_points_3d()
    labels = [
        "bottom_left","bottom_25","bottom_50","bottom_75",
        "bottom_right","right_25","right_50","right_75","top_right"
    ]
    K, dist = get_camera_matrix_and_dist(original.shape)

    # Homography (2D-2D)
    obj_2d = points_3d[:, :2].astype(np.float32)
    H, _ = cv2.findHomography(obj_2d, points_2d)
    proj_h = cv2.perspectiveTransform(obj_2d.reshape(-1,1,2), H).reshape(-1,2)

    # solvePnP (3D-2D)
    success, rvec, tvec = cv2.solvePnP(
        points_3d,
        points_2d,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    proj_pnp, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    proj_pnp = proj_pnp.reshape(-1,2)


    # Save final 2D image coordinates of the 9 key points
    points2d_txt_path = output_dir / f"{stem}_points2d.txt"
    with open(points2d_txt_path, "w") as f:
        for name, pt in zip(labels, points_2d):
            f.write(f"{name}: [{int(pt[0])}, {int(pt[1])}]\n")

    # Save images
    homography_img = draw_detected_and_projected(original, points_2d, proj_h, detected_color=(255,0,0), projected_color=(0,0,255))
    solvepnp_img = draw_detected_and_projected(original, points_2d, proj_pnp, detected_color=(255,0,0), projected_color=(0,0,255))
    homography_img_path = output_dir / f"{stem}_homography_proj.png"
    solvepnp_img_path = output_dir / f"{stem}_solvepnp_proj.png"
    cv2.imwrite(str(homography_img_path), homography_img)
    cv2.imwrite(str(solvepnp_img_path), solvepnp_img)

    # Print and save errors
    errors_h, mean_h = print_reprojection_errors(labels, points_2d, proj_h, "Homography")
    errors_pnp, mean_pnp = print_reprojection_errors(labels, points_2d, proj_pnp, "solvePnP")

    # Save errors to txt file
    error_txt_path = output_dir / f"{stem}_reprojection_errors.txt"
    with open(error_txt_path, "w") as f:
        f.write("Reprojection Error (Homography):\n")
        for name, err in zip(labels, errors_h):
            f.write(f"{name}: {err:.2f} px\n")
        f.write(f"Mean error: {mean_h:.2f} px\n\n")
        f.write("Reprojection Error (solvePnP):\n")
        for name, err in zip(labels, errors_pnp):
            f.write(f"{name}: {err:.2f} px\n")
        f.write(f"Mean error: {mean_pnp:.2f} px\n")



    stem = image_path.stem
    # Draw the blue segmentation mask overlay, then contour and points
    contour_annotated = original.copy()
    blue_mask = np.zeros_like(original)
    blue_mask[..., 0] = 255  # Blue channel
    mask_bool = mask_uint8 > 0
    alpha = 0.45
    # Overlay blue mask only where mask is present
    contour_annotated = cv2.addWeighted(contour_annotated, 1.0, blue_mask, alpha, 0, dtype=cv2.CV_8U)
    contour_annotated[~mask_bool] = original[~mask_bool]
    # Draw contour and points
    cv2.polylines(contour_annotated, [contour.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=3)  # Yellow
    for key in points:
        pt = points[key]
        cv2.circle(contour_annotated, tuple(pt.astype(int)), 6, (255, 0, 0), -1)  # Blue
    contour_annotated_path = output_dir / f"{stem}_contour_annotated.png"
    cv2.imwrite(str(contour_annotated_path), contour_annotated)

    # Centerline output (keep as before)
    point_keys = (
        "bottom_left",
        "bottom_right",
        "top_right",
        "bottom_50",
        "mirror_mount_left",
        "mirror_mount_right",
    )
    centerline = draw_points(original, points, point_keys)
    cv2.line(
        centerline,
        tuple(points["bottom_left"]),
        tuple(points["bottom_right"]),
        (0, 255, 0),
        3,
    )
    bottom_edge = points["bottom_right"] - points["bottom_left"]
    edge_height = np.linalg.norm(points["top_right"] - points["bottom_right"])
    perp_dir = np.array([-bottom_edge[1], bottom_edge[0]], dtype=np.float32)
    perp_norm = np.linalg.norm(perp_dir)
    if perp_norm == 0:
        raise RuntimeError("Bottom edge collapsed to a single point")
    perp_unit = perp_dir / perp_norm
    if perp_unit[1] > 0:
        perp_unit = -perp_unit
    center_start = points["bottom_50"]
    center_end = (center_start + perp_unit * edge_height).astype(np.int32)
    cv2.line(centerline, tuple(center_start), tuple(center_end), (0, 0, 255), 3)

    centerline_path = output_dir / f"{stem}_centerline.png"
    cv2.imwrite(str(centerline_path), centerline)

    contour_annotated_path = output_dir / f"{stem}_contour_annotated.png"
    cv2.imwrite(str(contour_annotated_path), contour_annotated)

    # No terminal printout for points or errors


if __name__ == "__main__":
    main()
