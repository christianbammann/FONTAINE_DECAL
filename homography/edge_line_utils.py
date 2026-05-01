import cv2
import numpy as np

from homography.line_math_utils import fit_line_from_points


def _pick_row_candidate_near_expected(row_points, expected_x, max_inward_px, max_outward_px):
    """choose a row sample near the expected door edge instead of blindly taking the furthest-right point"""
    row_points = np.array(row_points, dtype=np.float32)
    deltas = row_points[:, 0] - float(expected_x)

    preferred = row_points[(deltas >= -float(max_inward_px)) & (deltas <= float(max_outward_px))]
    if len(preferred) > 0:
        return preferred[np.argmax(preferred[:, 0])]

    closest_index = int(np.argmin(np.abs(deltas)))
    closest_delta = float(deltas[closest_index])
    fallback_limit = max(float(max_inward_px) * 1.75, float(max_outward_px) * 2.0)
    if abs(closest_delta) <= fallback_limit:
        return row_points[closest_index]
    return None


def fit_right_edge_from_contour(contour, roi, expected_point):
    """fit the outer right door edge directly from the segmentation contour"""
    x0, y0, x1, y1 = roi
    contour_float = contour.astype(np.float32)

    # keep only contour points inside the right-edge search box
    roi_points = contour_float[
        (contour_float[:, 0] >= x0) &
        (contour_float[:, 0] <= x1) &
        (contour_float[:, 1] >= y0) &
        (contour_float[:, 1] <= y1)
    ]
    if len(roi_points) < 8:
        raise RuntimeError("Not enough contour points in right-edge ROI")

    # for each image row, keep the outermost contour point on the right
    expected_x = float(expected_point[0])
    max_inward_px = 16.0
    max_outward_px = 8.0
    rows = np.rint(roi_points[:, 1]).astype(np.int32)
    unique_rows = np.unique(rows)
    collected = []
    for row in unique_rows:
        row_points = roi_points[rows == row]
        if len(row_points) == 0:
            continue
        candidate = _pick_row_candidate_near_expected(row_points, expected_x, max_inward_px, max_outward_px)
        if candidate is not None:
            collected.append(candidate)

    collected = np.array(collected, dtype=np.float32)
    if len(collected) < 8:
        raise RuntimeError("Not enough sampled contour rows for right-edge fit")

    # keep only points close to the expected x location to reject handle-pocket drift
    deltas = collected[:, 0] - expected_x
    collected = collected[(deltas >= -max_inward_px) & (deltas <= max_outward_px)]
    if len(collected) < 8:
        raise RuntimeError("Contour right-edge points drifted too far from expected location")

    # fit a line, then keep the inliers that stay near that fitted side edge
    fitted = fit_line_from_points(collected)
    direction = fitted[1] - fitted[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    deltas = collected - fitted[0]
    distances = np.abs(deltas[:, 0] * direction[1] - deltas[:, 1] * direction[0])
    distance_limit = max(3.0, float(np.percentile(distances, 70)) * 1.5)
    inliers = collected[distances <= distance_limit]
    if len(inliers) < 8:
        raise RuntimeError("Right-edge contour inliers were too sparse")

    return fit_line_from_points(inliers)


def fit_right_edge_from_contour_midsection(contour, roi, top_right, bottom_right):
    """fit the straighter lower-right section of the segmented door edge, avoiding the noisy upper area"""
    x0, y0, x1, y1 = roi
    contour_float = contour.astype(np.float32)

    y_top = float(min(top_right[1], bottom_right[1]))
    y_bottom = float(max(top_right[1], bottom_right[1]))
    span = max(1.0, y_bottom - y_top)

    # Use a quieter band in the lower part of the door side rather than the broad middle.
    # This avoids the upper noise and reduces the odds of locking onto nearby external seams.
    lower_band_start_fraction = 0.10
    lower_band_end_fraction = 0.38
    band_y0 = max(float(y0), y_bottom - lower_band_end_fraction * span)
    band_y1 = min(float(y1), y_bottom - lower_band_start_fraction * span)
    if band_y1 <= band_y0:
        raise RuntimeError("Right-edge lower band collapsed")

    roi_points = contour_float[
        (contour_float[:, 0] >= x0) &
        (contour_float[:, 0] <= x1) &
        (contour_float[:, 1] >= band_y0) &
        (contour_float[:, 1] <= band_y1)
    ]
    if len(roi_points) < 8:
        raise RuntimeError("Not enough contour points in right-edge lower band")

    expected_x = float(0.5 * (float(top_right[0]) + float(bottom_right[0])))
    max_inward_px = max(12.0, min(18.0, 0.10 * span))
    max_outward_px = max(6.0, min(9.0, 0.045 * span))
    rows = np.rint(roi_points[:, 1]).astype(np.int32)
    unique_rows = np.unique(rows)
    collected = []
    for row in unique_rows:
        row_points = roi_points[rows == row]
        if len(row_points) == 0:
            continue
        candidate = _pick_row_candidate_near_expected(row_points, expected_x, max_inward_px, max_outward_px)
        if candidate is not None:
            collected.append(candidate)

    collected = np.array(collected, dtype=np.float32)
    if len(collected) < 8:
        raise RuntimeError("Not enough sampled rows in right-edge lower band")

    # Stay close to the expected door edge so nearby seams do not win just because they are farther right.
    deltas = collected[:, 0] - expected_x
    collected = collected[(deltas >= -max_inward_px) & (deltas <= max_outward_px)]
    if len(collected) < 8:
        raise RuntimeError("Right-edge lower band drifted too far from expected x")

    fitted = fit_line_from_points(collected)
    direction = fitted[1] - fitted[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    deltas = collected - fitted[0]
    distances = np.abs(deltas[:, 0] * direction[1] - deltas[:, 1] * direction[0])
    distance_limit = max(2.5, float(np.percentile(distances, 70)) * 1.4)
    inliers = collected[distances <= distance_limit]
    if len(inliers) < 8:
        raise RuntimeError("Right-edge lower band inliers were too sparse")

    return fit_line_from_points(inliers)


def line_x_at_y(line, y):
    """x position of a line at a given y"""
    p1 = line[0].astype(np.float32)
    p2 = line[1].astype(np.float32)
    direction = p2 - p1
    if abs(float(direction[1])) < 1e-6:
        raise RuntimeError("Cannot evaluate x at y for a horizontal line")
    return float(p1[0] + (float(y) - p1[1]) * direction[0] / direction[1])


def shift_line_horizontally(line, shift_x):
    """move a line left or right without changing its angle"""
    delta = np.array([float(shift_x), 0.0], dtype=np.float32)
    return line[0].astype(np.float32) + delta, line[1].astype(np.float32) + delta


def refine_right_line_by_global_offset(
    gray,
    boundary_mask,
    roi,
    prior_line,
    inward_search_px=8,
    outward_search_px=14,
    max_inward_shift=6.0,
    max_outward_shift=12.0,
):
    """estimate one robust global shift from the contour right-line prior to the visible edge"""
    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    boundary_patch = boundary_mask[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError("Empty ROI for right-line global offset refinement")

    blur = cv2.GaussianBlur(patch, (5, 5), 0)
    half_window = 5
    shift_values = np.arange(-int(max_inward_shift), int(max_outward_shift) + 1, dtype=np.int32)

    best_shift = None
    best_score = None

    for shift in shift_values:
        row_scores = []

        for local_row in range(half_window, blur.shape[0] - half_window, 2):
            y = y0 + local_row
            prior_x = line_x_at_y(prior_line, y)
            col = int(round(prior_x - x0 + shift))

            if col < half_window or col >= blur.shape[1] - half_window:
                continue

            boundary_cols = np.where(boundary_patch[local_row] > 0)[0]
            if len(boundary_cols) > 0:
                boundary_col = int(np.max(boundary_cols))
                if col < boundary_col - inward_search_px or col > boundary_col + outward_search_px:
                    continue
                boundary_penalty = 0.35 * abs(float(col - boundary_col))
            else:
                boundary_penalty = 0.10 * abs(float(shift))

            left_mean = float(np.mean(blur[local_row, col - half_window : col]))
            right_mean = float(np.mean(blur[local_row, col + 1 : col + 1 + half_window]))
            contrast = abs(left_mean - right_mean)

            # reward strong edge evidence while keeping the line close to the contour prior
            row_scores.append(contrast - boundary_penalty)

        if len(row_scores) < 12:
            continue

        row_scores = np.array(row_scores, dtype=np.float32)
        median_score = float(np.median(row_scores))
        top_half_mean = float(np.mean(np.sort(row_scores)[len(row_scores) // 2 :]))
        total_score = median_score + 0.35 * top_half_mean - 0.08 * abs(float(shift))

        if best_score is None or total_score > best_score:
            best_score = total_score
            best_shift = int(shift)

    if best_shift is None:
        raise RuntimeError("Could not find a stable global offset for the right line")

    return shift_line_horizontally(prior_line, best_shift)


def detect_outer_right_line_from_image(
    gray,
    boundary_mask,
    roi,
    expected_point,
    inward_tolerance=10,
    outward_tolerance=18,
):
    """fit the outer visible right edge near the segmentation boundary, not arbitrary background edges"""
    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    boundary_patch = boundary_mask[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError("Empty ROI for outer right-edge detection")

    edges = cv2.Canny(cv2.GaussianBlur(patch, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    right_points = []
    expected_local_x = float(expected_point[0] - x0)

    for local_row in range(edges.shape[0]):
        cols = np.where(edges[local_row] > 0)[0]
        if len(cols) == 0:
            continue

        boundary_cols = np.where(boundary_patch[local_row] > 0)[0]
        if len(boundary_cols) == 0:
            continue

        boundary_col = int(np.max(boundary_cols))
        left_limit = max(0, boundary_col - inward_tolerance)
        right_limit = min(edges.shape[1] - 1, boundary_col + outward_tolerance)

        row_candidates = cols[(cols >= left_limit) & (cols <= right_limit)]
        if len(row_candidates) == 0:
            # fallback to expected x if the mask boundary row is weak
            left_limit = max(0, int(round(expected_local_x - inward_tolerance)))
            right_limit = min(edges.shape[1] - 1, int(round(expected_local_x + outward_tolerance)))
            row_candidates = cols[(cols >= left_limit) & (cols <= right_limit)]

        if len(row_candidates) == 0:
            continue

        # prefer the strongest local contrast near the segmentation boundary
        scored = []
        for col in row_candidates:
            left0 = max(0, int(col) - 6)
            left1 = max(left0 + 1, int(col))
            right0 = min(patch.shape[1] - 1, int(col) + 1)
            right1 = min(patch.shape[1], int(col) + 7)
            if right1 <= right0:
                continue

            left_mean = float(np.mean(patch[local_row, left0:left1]))
            right_mean = float(np.mean(patch[local_row, right0:right1]))
            contrast = abs(left_mean - right_mean)
            deltas = float(col) - float(boundary_col)

            score = 2.0 * contrast - 1.4 * abs(deltas) + 0.12 * deltas
            scored.append((score, int(col), contrast, deltas))

        if not scored:
            continue

        best_score = max(entry[0] for entry in scored)
        near_best = [entry for entry in scored if entry[0] >= best_score - 2.0]

        # among similarly good candidates, prefer a small outward move rather than the most inward one
        outward_near_best = [entry for entry in near_best if entry[3] >= -1.0]
        if outward_near_best:
            near_best = outward_near_best

        best_col = max(near_best, key=lambda entry: entry[1])[1]

        right_points.append([float(x0 + best_col), float(y0 + local_row)])

    if len(right_points) < 8:
        raise RuntimeError("Not enough outer right-edge points found in image ROI")

    fitted = fit_line_from_points(np.array(right_points, dtype=np.float32))
    return resnap_line_to_visible_edge(gray, roi, fitted, "vertical", prefer_direction=1, search_radius=28)


def detect_dominant_line(gray, boundary_mask, mask_inside, roi, orientation, expected_point):
    """backup way to find a line with hough"""
    # unpack the roi box
    x0, y0, x1, y1 = roi

    # crop image and masks to this box
    patch = gray[y0:y1, x0:x1]
    boundary_patch = boundary_mask[y0:y1, x0:x1]
    inside_patch = mask_inside[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError(f"Empty ROI for {orientation} line detection")

    # find strong edges in this patch
    edges = cv2.Canny(cv2.GaussianBlur(patch, (5, 5), 0), 50, 150)

    # keep only edges on the door border
    edges = cv2.bitwise_and(edges, boundary_patch)

    # ask hough for straight lines
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

    # rough place where line should be
    expected = expected_point.astype(np.float32)
    best_score = None
    best_line = None

    # score each found line and keep best one
    for entry in lines[:, 0]:
        # move local line points back to full image
        x_start, y_start, x_end, y_end = entry.astype(np.float32)
        p1 = np.array([x_start + x0, y_start + y0], dtype=np.float32)
        p2 = np.array([x_end + x0, y_end + y0], dtype=np.float32)

        # line length
        delta = p2 - p1
        length = float(np.linalg.norm(delta))
        if length < 20:
            continue

        # check if line angle fits what we want
        dx = abs(float(delta[0]))
        dy = abs(float(delta[1]))
        if orientation == "horizontal" and dx < dy * 2.0:
            continue
        if orientation == "vertical" and dy < dx * 1.3:
            continue

        # score likes long lines near expected spot
        score = length - 1.5 * float(np.linalg.norm((p1 + p2) / 2.0 - expected))

        if orientation == "vertical":
            # for vertical line also check that it stays inside the door
            inside_hits = 0
            valid_hits = 0
            for t in np.linspace(0.0, 1.0, 25, dtype=np.float32):
                sample = p1 + t * delta
                sx = int(round(sample[0])) - x0
                sy = int(round(sample[1])) - y0

                # only count samples inside this roi
                if 0 <= sx < inside_patch.shape[1] and 0 <= sy < inside_patch.shape[0]:
                    valid_hits += 1
                    if inside_patch[sy, sx] > 0:
                        inside_hits += 1
            if valid_hits == 0:
                continue
            score += 80.0 * (inside_hits / valid_hits)

        # save best line so far
        if best_score is None or score > best_score:
            best_score = score
            best_line = (p1, p2)

    if best_line is None:
        raise RuntimeError(f"No usable {orientation} line candidates found")
    return best_line


def resnap_line_to_visible_edge(gray, roi, line, orientation, prefer_direction, search_radius=24):
    """keep a line's angle but slide it onto the visible outer edge"""
    x0, y0, x1, y1 = roi
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError(f"Empty ROI for {orientation} line resnap")

    edges = cv2.Canny(cv2.GaussianBlur(patch, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    p1 = line[0].astype(np.float32)
    p2 = line[1].astype(np.float32)
    direction = p2 - p1
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise RuntimeError("Cannot resnap a zero-length line")

    points = []

    if orientation == "vertical":
        if abs(float(direction[1])) < 1e-6:
            raise RuntimeError("Vertical line fit became horizontal during resnap")

        for row in range(max(0, y0), min(gray.shape[0], y1)):
            y = float(row)
            x_on_line = float(p1[0] + (y - p1[1]) * direction[0] / direction[1])
            local_row = row - y0
            start = int(round(x_on_line))

            if prefer_direction > 0:
                candidates = range(start, min(x1 - 1, start + search_radius) + 1)
            else:
                candidates = range(start, max(x0, start - search_radius) - 1, -1)

            for col in candidates:
                local_col = col - x0
                if 0 <= local_col < edges.shape[1] and edges[local_row, local_col] > 0:
                    points.append([col, row])
                    break

    elif orientation == "horizontal":
        if abs(float(direction[0])) < 1e-6:
            raise RuntimeError("Horizontal line fit became vertical during resnap")

        for col in range(max(0, x0), min(gray.shape[1], x1)):
            x = float(col)
            y_on_line = float(p1[1] + (x - p1[0]) * direction[1] / direction[0])
            local_col = col - x0
            start = int(round(y_on_line))

            if prefer_direction > 0:
                candidates = range(start, min(y1 - 1, start + search_radius) + 1)
            else:
                candidates = range(start, max(y0, start - search_radius) - 1, -1)

            for row in candidates:
                local_row = row - y0
                if 0 <= local_row < edges.shape[0] and edges[local_row, local_col] > 0:
                    points.append([col, row])
                    break
    else:
        raise RuntimeError(f"Unsupported orientation for resnap: {orientation}")

    if len(points) < 8:
        raise RuntimeError(f"Not enough visible-edge points to resnap {orientation} line")

    return fit_line_from_points(np.array(points, dtype=np.float32))


def sample_inside_boundary_points(gray, mask_inside, boundary_mask, roi, orientation):
    """grab edge points from the door border"""
    # unpack the roi box
    x0, y0, x1, y1 = roi

    # crop image and masks to this box
    patch = gray[y0:y1, x0:x1]
    inside_patch = mask_inside[y0:y1, x0:x1]
    boundary_patch = boundary_mask[y0:y1, x0:x1]
    if patch.size == 0:
        raise RuntimeError(f"Empty ROI for {orientation} point sampling")

    # find edges and keep only border edges
    edges = cv2.Canny(cv2.GaussianBlur(patch, (5, 5), 0), 50, 150)
    candidate = cv2.bitwise_and(edges, boundary_patch)

    # make edge a little thicker so finding points is easier
    candidate = cv2.dilate(candidate, np.ones((3, 3), np.uint8), iterations=1)

    # save found points here
    collected = []

    # for bottom line scan each column
    if orientation == "horizontal":
        for col in range(candidate.shape[1]):
            # edge pixels in this column
            rows = np.where(candidate[:, col] > 0)[0]
            for row in np.sort(rows):
                # check if area above is still inside the door
                top = max(0, row - 3)
                bottom = min(inside_patch.shape[0], row + 2)
                if np.any(inside_patch[top:bottom, col] > 0):
                    collected.append([x0 + col, y0 + row])
                    break
    # for right line scan each row
    elif orientation == "vertical":
        for row in range(candidate.shape[0]):
            # edge pixels in this row
            cols = np.where(candidate[row] > 0)[0]
            for col in np.sort(cols):
                # check if area left and right of edge is still inside the door
                left = max(0, col - 2)
                right = min(inside_patch.shape[1], col + 3)
                if np.any(inside_patch[row, left:right] > 0):
                    collected.append([x0 + col, y0 + row])
                    break
    else:
        raise RuntimeError(f"Unsupported orientation: {orientation}")

    # error if too few points
    if len(collected) < 8:
        raise RuntimeError(f"Not enough sampled points for {orientation} line fit")

    # return points as numpy array
    return np.array(collected, dtype=np.float32)


def refine_line(gray, boundary_mask, mask_inside, roi, orientation, expected_point):
    """try easy line fit first then backup line fit"""
    try:
        # first try sampling border points then fit line
        points = sample_inside_boundary_points(gray, mask_inside, boundary_mask, roi, orientation)
        return fit_line_from_points(points)
    except RuntimeError:
        # if that fails use hough backup
        return detect_dominant_line(gray, boundary_mask, mask_inside, roi, orientation, expected_point)
