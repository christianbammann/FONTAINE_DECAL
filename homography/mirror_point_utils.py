import cv2
import numpy as np


def fit_line_endpoints(points: np.ndarray):
    """fit a line through points and return extreme endpoints along that line"""
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([vx, vy], dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    center = np.array([x0, y0], dtype=np.float32)

    deltas = points.astype(np.float32) - center
    along = deltas[:, 0] * direction[0] + deltas[:, 1] * direction[1]
    start = center + float(np.min(along)) * direction
    end = center + float(np.max(along)) * direction
    return start, end


def choose_best_mirror_blob(
    search_contours,
    search_x0: int,
    search_y0: int,
    x_min: int,
    y_min: int,
    door_width: int,
    door_height: int,
):
    """pick the dark blob that looks most like the mirror mount"""
    expected = np.array([x_min + 0.09 * door_width, y_min + 0.30 * door_height], dtype=np.float32)
    best_contour = None
    best_score = None

    for candidate in search_contours:
        area = cv2.contourArea(candidate)
        if area < 2500:
            continue

        x, y, w, h = cv2.boundingRect(candidate)
        global_x = x + search_x0
        global_y = y + search_y0

        if h < 120 or w < 50:
            continue
        if global_x > x_min + 0.20 * door_width:
            continue
        if global_y < y_min + 0.10 * door_height or global_y > y_min + 0.48 * door_height:
            continue

        center = np.array([global_x + w / 2.0, global_y + h / 2.0], dtype=np.float32)
        score = area + 4.0 * h - 18.0 * float(np.linalg.norm(center - expected))
        if best_score is None or score > best_score:
            best_score = score
            best_contour = candidate

    return best_contour


def select_lower_hull_edge(hull_points: np.ndarray, bx: int, by: int, bw: int, bh: int):
    """choose the lower slanted hull edge that forms the mirror base"""
    hull_contour = hull_points.astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(hull_contour, True))
    simple = cv2.approxPolyDP(hull_contour, 0.010 * perimeter, True).reshape(-1, 2).astype(np.float32)

    best_edge = None
    best_score = None
    min_mid_y = by + 0.52 * bh

    for index in range(len(simple)):
        start = simple[index]
        end = simple[(index + 1) % len(simple)]
        if start[0] <= end[0]:
            left = start
            right = end
        else:
            left = end
            right = start

        delta = right - left
        length = float(np.linalg.norm(delta))
        if length < max(28.0, 0.22 * bw):
            continue

        dx = float(delta[0])
        dy = float(delta[1])
        if dx <= 0.0:
            continue
        if abs(dx) / max(length, 1e-6) < 0.86:
            continue

        # image y grows downward, so a good mirror-base edge typically rises slightly to the right
        if dy > max(12.0, 0.10 * bh):
            continue
        if dy < -max(34.0, 0.28 * bh):
            continue

        mid_x = float((left[0] + right[0]) / 2.0)
        mid_y = float((left[1] + right[1]) / 2.0)
        if mid_y < min_mid_y:
            continue

        if left[0] > bx + 0.22 * bw:
            continue
        if right[0] < bx + 0.68 * bw:
            continue

        score = (
            4.0 * length
            + 1.4 * (mid_y - by)
            + 1.2 * (right[0] - (bx + 0.68 * bw))
            + 0.6 * ((bx + 0.18 * bw) - left[0])
            - 0.8 * abs(dy)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_edge = (left, right)

    return best_edge


def select_lower_contour_edge(contour_points: np.ndarray, bx: int, by: int, bw: int, bh: int):
    """choose the lower slanted contour edge that forms the mirror base"""
    contour = contour_points.astype(np.float32).reshape(-1, 1, 2)
    simple = cv2.approxPolyDP(contour, 0.006 * cv2.arcLength(contour, True), True).reshape(-1, 2).astype(np.float32)

    best_edge = None
    best_score = None
    lower_y = by + 0.58 * bh

    for index in range(len(simple)):
        start = simple[index]
        end = simple[(index + 1) % len(simple)]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < max(20.0, 0.18 * bw):
            continue

        if float((start[1] + end[1]) / 2.0) < lower_y:
            continue
        if abs(float(delta[0])) / max(length, 1e-6) < 0.80:
            continue

        if start[0] <= end[0]:
            left = start
            right = end
        else:
            left = end
            right = start
        delta_lr = right - left

        if float(delta_lr[1]) < -max(24.0, 0.22 * bh):
            continue
        if float(delta_lr[1]) > max(18.0, 0.14 * bh):
            continue

        mid_y = float((left[1] + right[1]) / 2.0)
        score = (
            2.2 * length
            + 1.4 * (mid_y - by)
            + 0.35 * (float(right[0]) - (bx + 0.55 * bw))
            - 0.20 * abs(float(left[0]) - bx)
        )
        if best_score is None or score > best_score:
            best_score = score
            best_edge = (left, right)

    return best_edge


def detect_lower_edge_from_silhouette(contour_points: np.ndarray, image_shape, bx: int, by: int, bw: int, bh: int):
    """find the long lower mirror-base edge from the blob's lower silhouette"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour_points.astype(np.int32).reshape(-1, 1, 2)], -1, 255, thickness=cv2.FILLED)

    xs = []
    ys = []
    for x in range(int(bx), int(bx + bw)):
        column = np.where(mask[:, x] > 0)[0]
        if len(column):
            xs.append(float(x))
            ys.append(float(np.max(column)))

    if len(xs) < max(20, int(0.18 * bw)):
        return None

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    kernel_size = int(max(9, min(31, 2 * int(0.03 * bw) + 1)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    ys_smooth = np.convolve(ys, kernel, mode="same")
    slope = np.gradient(ys_smooth)

    candidate_mask = (
        (ys_smooth >= by + 0.55 * bh)
        & (slope >= -1.0)
        & (slope <= 0.2)
    )

    runs = []
    start = None
    for index, is_candidate in enumerate(candidate_mask):
        if is_candidate and start is None:
            start = index
        if (not is_candidate) and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(candidate_mask) - 1))

    best_result = None
    best_score = None

    for start, end in runs:
        x_span = xs[end] - xs[start]
        if x_span < max(30.0, 0.18 * bw):
            continue

        segment = np.column_stack([xs[start : end + 1], ys[start : end + 1]]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(segment.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        direction = np.array([vx, vy], dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        center = np.array([x0, y0], dtype=np.float32)
        residual = float(np.mean(np.abs((segment - center) @ normal)))

        score = 4.0 * x_span - 40.0 * residual + 0.2 * float(np.mean(segment[:, 1]))
        if best_score is None or score > best_score:
            best_score = score
            best_result = (start, end, center, direction)

    if best_result is None:
        return None

    start, end, center, direction = best_result
    if abs(float(direction[0])) < 1e-6:
        return None

    left_x = xs[start]
    right_x = xs[end]
    t_left = (left_x - center[0]) / direction[0]
    t_right = (right_x - center[0]) / direction[0]
    left_point = center + t_left * direction
    right_point = center + t_right * direction

    if left_point[0] > right_point[0]:
        left_point, right_point = right_point, left_point

    return np.rint(left_point).astype(np.int32), np.rint(right_point).astype(np.int32)


def refine_edge_with_raw_points(raw_points: np.ndarray, left_seed: np.ndarray, right_seed: np.ndarray):
    """extend a seed edge to the true endpoints supported by raw points"""
    seed_left = left_seed.astype(np.float32)
    seed_right = right_seed.astype(np.float32)
    direction = seed_right - seed_left
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.rint(seed_left).astype(np.int32), np.rint(seed_right).astype(np.int32)

    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    center = 0.5 * (seed_left + seed_right)
    deltas = raw_points.astype(np.float32) - center
    along = deltas[:, 0] * direction[0] + deltas[:, 1] * direction[1]
    across = np.abs(deltas[:, 0] * normal[0] + deltas[:, 1] * normal[1])

    support_mask = (
        (along >= -0.65 * length)
        & (along <= 0.65 * length)
        & (across <= max(4.0, 0.05 * length))
    )
    support = raw_points[support_mask].astype(np.float32)
    if len(support) < 2:
        return np.rint(seed_left).astype(np.int32), np.rint(seed_right).astype(np.int32)

    support_deltas = support - center
    support_along = support_deltas[:, 0] * direction[0] + support_deltas[:, 1] * direction[1]
    left_point = support[int(np.argmin(support_along))]
    right_point = support[int(np.argmax(support_along))]

    if left_point[0] > right_point[0]:
        left_point, right_point = right_point, left_point

    return np.rint(left_point).astype(np.int32), np.rint(right_point).astype(np.int32)


def fallback_lower_edge_from_hull(hull_points: np.ndarray, bx: int, by: int, bw: int, bh: int):
    """fallback if a single simplified hull edge is not available"""
    lower_band = hull_points[
        (hull_points[:, 1] >= by + 0.48 * bh) &
        (hull_points[:, 0] >= bx - 4) &
        (hull_points[:, 0] <= bx + bw + 4)
    ]
    if len(lower_band) < 6:
        return None

    start, end = fit_line_endpoints(lower_band.astype(np.float32))
    if start[0] <= end[0]:
        left_seed = start
        right_seed = end
    else:
        left_seed = end
        right_seed = start

    delta = right_seed - left_seed
    length = float(np.linalg.norm(delta))
    if length < max(28.0, 0.22 * bw):
        return None
    if abs(float(delta[0])) / max(length, 1e-6) < 0.84:
        return None

    return refine_edge_with_raw_points(hull_points, left_seed, right_seed)


def refine_left_tip_with_image_corner(gray: np.ndarray, left_point: np.ndarray, right_point: np.ndarray):
    """pull the left mirror point onto the visible door-side start of the mirror base"""
    left = left_point.astype(np.float32)
    right = right_point.astype(np.float32)
    direction = right - left
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.rint(left).astype(np.int32)

    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    x0 = max(0, int(left[0] - max(35.0, 0.06 * length)))
    x1 = min(gray.shape[1], int(left[0] + max(85.0, 0.15 * length)))
    y0 = max(0, int(left[1] - max(70.0, 0.12 * length)))
    y1 = min(gray.shape[0], int(left[1] + max(70.0, 0.12 * length)))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.rint(left).astype(np.int32)

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    corners = cv2.goodFeaturesToTrack(
        roi_blur,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.rint(left).astype(np.int32)

    def sample(point: np.ndarray):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = max(0, min(gray.shape[1] - 1, x))
        y = max(0, min(gray.shape[0] - 1, y))
        return float(gray[y, x])

    best_point = left.copy()
    best_score = None

    for corner in corners.reshape(-1, 2):
        point = np.array([corner[0] + x0, corner[1] + y0], dtype=np.float32)
        delta = point - left
        along = float(delta[0] * direction[0] + delta[1] * direction[1])
        cross = abs(float(delta[0] * normal[0] + delta[1] * normal[1]))

        if along < -8.0:
            continue
        if point[0] < left[0] - 6.0:
            continue
        if point[0] > left[0] + max(28.0, 0.08 * length):
            continue
        if cross > max(14.0, 0.05 * length):
            continue
        if abs(float(point[1] - left[1])) > max(28.0, 0.08 * length):
            continue

        # the correct left endpoint has dark mirror interior above/right of it,
        # with brighter background/gap to the left and brighter door below.
        inside = sample(point - 6.0 * normal + 4.0 * direction)
        outside = sample(point - 10.0 * direction)
        below = sample(point + 10.0 * normal)
        outside_contrast = outside - inside
        below_contrast = below - inside

        if outside_contrast < 55.0:
            continue
        if below_contrast < 40.0:
            continue

        score = (
            0.8 * outside_contrast
            + 0.7 * below_contrast
            + 0.6 * max(along, 0.0)
            - 0.5 * cross
            - 0.15 * abs(float(point[1] - left[1]))
        )
        if best_score is None or score > best_score:
            best_score = score
            best_point = point

    return np.rint(best_point).astype(np.int32)


def refine_right_tip_with_image_corner(gray: np.ndarray, left_point: np.ndarray, right_point: np.ndarray):
    """push the right mirror point to the real image corner near the tip"""
    left = left_point.astype(np.float32)
    right = right_point.astype(np.float32)
    direction = right - left
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.rint(right).astype(np.int32)

    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    x0 = max(0, int(right[0] - max(40.0, 0.08 * length)))
    x1 = min(gray.shape[1], int(right[0] + max(90.0, 0.18 * length)))
    y0 = max(0, int(right[1] - max(90.0, 0.18 * length)))
    y1 = min(gray.shape[0], int(right[1] + max(55.0, 0.11 * length)))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.rint(right).astype(np.int32)

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    corners = cv2.goodFeaturesToTrack(
        roi_blur,
        maxCorners=40,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.rint(right).astype(np.int32)

    def sample(point: np.ndarray):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = max(0, min(gray.shape[1] - 1, x))
        y = max(0, min(gray.shape[0] - 1, y))
        return float(gray[y, x])

    best_point = right.copy()
    best_score = None

    for corner in corners.reshape(-1, 2):
        point = np.array([corner[0] + x0, corner[1] + y0], dtype=np.float32)
        delta = point - right
        forward = float(delta[0] * direction[0] + delta[1] * direction[1])
        cross = abs(float(delta[0] * normal[0] + delta[1] * normal[1]))

        if forward < -4.0:
            continue
        if point[0] < right[0] - 2.0:
            continue
        if cross > max(12.0, 0.03 * length):
            continue
        if abs(float(point[1] - right[1])) > max(24.0, 0.06 * length):
            continue

        # true tip corners still have dark mirror interior up-left of the point
        inside = sample(point - 8.0 * normal - 4.0 * direction)
        below = sample(point + 10.0 * normal)
        ahead = sample(point + 10.0 * direction)
        below_contrast = below - inside
        ahead_contrast = ahead - inside

        if below_contrast < 35.0:
            continue
        if ahead_contrast < 25.0:
            continue

        score = (
            1.4 * (point[0] - right[0])
            + 0.7 * forward
            + 0.8 * below_contrast
            + 0.5 * ahead_contrast
            - 0.7 * cross
            - 0.15 * abs(float(point[1] - right[1]))
        )
        if best_score is None or score > best_score:
            best_score = score
            best_point = point

    return np.rint(best_point).astype(np.int32)


def detect_mirror_points(gray: np.ndarray, contour: np.ndarray):
    """find left and right points on the lower slanted mirror-mount edge"""
    x_min = int(np.min(contour[:, 0]))
    x_max = int(np.max(contour[:, 0]))
    y_min = int(np.min(contour[:, 1]))
    y_max = int(np.max(contour[:, 1]))
    door_width = max(1, x_max - x_min)
    door_height = max(1, y_max - y_min)

    search = gray[
        max(0, y_min + int(0.12 * door_height)) : min(gray.shape[0], y_min + int(0.68 * door_height)),
        max(0, x_min - int(0.10 * door_width)) : min(gray.shape[1], x_min + int(0.30 * door_width)),
    ]
    if search.size == 0:
        return {}

    search_blur = cv2.GaussianBlur(search, (5, 5), 0)
    _, dark = cv2.threshold(search_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    search_contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not search_contours:
        return {}

    search_x0 = max(0, x_min - int(0.10 * door_width))
    search_y0 = max(0, y_min + int(0.12 * door_height))
    best_contour = choose_best_mirror_blob(
        search_contours,
        search_x0,
        search_y0,
        x_min,
        y_min,
        door_width,
        door_height,
    )
    if best_contour is None:
        return {}

    bx, by, bw, bh = cv2.boundingRect(best_contour)
    global_contour = best_contour + np.array([[[search_x0, search_y0]]], dtype=np.int32)
    contour_points = global_contour.reshape(-1, 2).astype(np.float32)
    hull_points = cv2.convexHull(global_contour).reshape(-1, 2).astype(np.float32)

    silhouette_edge = detect_lower_edge_from_silhouette(
        contour_points,
        gray.shape,
        bx + search_x0,
        by + search_y0,
        bw,
        bh,
    )

    if silhouette_edge is not None:
        left_point, right_point = silhouette_edge
    else:
        best_edge = select_lower_contour_edge(
            contour_points,
            bx + search_x0,
            by + search_y0,
            bw,
            bh,
        )

        if best_edge is not None:
            left_point, right_point = refine_edge_with_raw_points(
                contour_points,
                best_edge[0],
                best_edge[1],
            )
        else:
            best_hull_edge = select_lower_hull_edge(
                hull_points,
                bx + search_x0,
                by + search_y0,
                bw,
                bh,
            )
            if best_hull_edge is not None:
                left_point, right_point = refine_edge_with_raw_points(
                    hull_points,
                    best_hull_edge[0],
                    best_hull_edge[1],
                )
            else:
                fallback = fallback_lower_edge_from_hull(
                    hull_points,
                    bx + search_x0,
                    by + search_y0,
                    bw,
                    bh,
                )
                if fallback is None:
                    return {}
                left_point, right_point = fallback

    left_point = refine_left_tip_with_image_corner(gray, left_point, right_point)
    right_point = refine_right_tip_with_image_corner(gray, left_point, right_point)

    return {
        "mirror_mount_left": left_point,
        "mirror_mount_right": right_point,
        "mirror_mount_corner": right_point,
    }
