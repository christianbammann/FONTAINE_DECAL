import cv2
import numpy as np


def line_coeffs_from_points(p1: np.ndarray, p2: np.ndarray):
    """turn 2 points into line equation numbers"""
    # pull out x and y values
    x1, y1 = p1.astype(np.float64)
    x2, y2 = p2.astype(np.float64)

    # for ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def intersect_lines(first, second):
    """find where 2 lines cross"""
    # turn both lines into equation form
    a1, b1, c1 = line_coeffs_from_points(first[0], first[1])
    a2, b2, c2 = line_coeffs_from_points(second[0], second[1])

    # if det is too small the lines are almost parallel
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        raise RuntimeError("The fitted door lines are too parallel to intersect")

    # solve for x and y intersection
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def project_point_to_line(point: np.ndarray, line):
    """push a point onto a line"""
    # get the 2 line points
    p1 = line[0].astype(np.float32)
    p2 = line[1].astype(np.float32)

    # direction of the line
    direction = p2 - p1

    # line length squared
    length_sq = float(np.dot(direction, direction))
    if length_sq == 0:
        raise RuntimeError("Cannot project onto a zero-length line")

    # how far to move along the line
    amount = float(np.dot(point.astype(np.float32) - p1, direction) / length_sq)
    return p1 + amount * direction


def point_line_distance(point: np.ndarray, line):
    """perpendicular distance from one point to one line"""
    p1 = line[0].astype(np.float32)
    p2 = line[1].astype(np.float32)
    direction = p2 - p1
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise RuntimeError("Cannot measure distance to a zero-length line")

    delta = point.astype(np.float32) - p1
    return abs(float(delta[0] * direction[1] - delta[1] * direction[0])) / norm


def fit_line_from_points(points: np.ndarray):
    """fit one long line through points"""
    # need at least 2 points
    if len(points) < 2:
        raise RuntimeError("Need at least two points to fit a line")

    # cv2.fitLine needs a certain line shape
    pts = points.astype(np.float32).reshape(-1, 1, 2)

    # fit the best line
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)

    # line direction and center point
    direction = np.array([vx, vy], dtype=np.float32)
    center = np.array([x0, y0], dtype=np.float32)

    # extend the line for visibility
    return np.array([center - 1000.0 * direction, center + 1000.0 * direction], dtype=np.float32)


def extend_line_across_width(line: np.ndarray, image_width: int):
    """rebuild a line so it spans the full image width"""
    p1 = np.array(line[0], dtype=np.float32)
    p2 = np.array(line[1], dtype=np.float32)
    direction = p2 - p1
    if abs(float(direction[0])) < 1e-6:
        return np.array([p1, p2], dtype=np.float32)

    x_left = 0.0
    x_right = float(max(0, int(image_width) - 1))
    y_left = float(p1[1] + (x_left - p1[0]) * direction[1] / direction[0])
    y_right = float(p1[1] + (x_right - p1[0]) * direction[1] / direction[0])
    return np.array([[x_left, y_left], [x_right, y_right]], dtype=np.float32)


def snap_point_to_contour(point, contour, max_dist=25.0):
    """move a point to the nearest contour point if it is close enough"""
    # work in float for distance math
    target = point.astype(np.float32)
    contour_float = contour.astype(np.float32)

    # distance from target to every contour point
    deltas = contour_float - target
    dist_sq = np.sum(deltas * deltas, axis=1)
    best_index = int(np.argmin(dist_sq))
    best_point = contour_float[best_index]

    # if nearest contour point is too far keep original point
    if float(np.sqrt(dist_sq[best_index])) > max_dist:
        return target
    return best_point


def snap_top_right_corner(point, contour, x_band=25.0, y_band=25.0, max_dist=50.0):
    """snap top right to points near the true top right corner area"""
    # work in float for distance math
    target = point.astype(np.float32)
    contour_float = contour.astype(np.float32)

    # top right corner should live near the max x and min y of the contour
    max_x = float(np.max(contour_float[:, 0]))
    min_y = float(np.min(contour_float[:, 1]))

    # first try the actual top right corner zone
    corner_zone = contour_float[
        (contour_float[:, 0] >= max_x - x_band) &
        (contour_float[:, 1] <= min_y + y_band)
    ]

    # if that is empty fall back to points near the guess
    if len(corner_zone) == 0:
        corner_zone = contour_float[
            (np.abs(contour_float[:, 0] - target[0]) <= max_dist) &
            (np.abs(contour_float[:, 1] - target[1]) <= max_dist)
        ]

    # if still empty just use the normal snap
    if len(corner_zone) == 0:
        return snap_point_to_contour(point, contour, max_dist=max_dist)

    # prefer the true outer corner, not the inside-biased line guess
    deltas = corner_zone - target
    dist_sq = np.sum(deltas * deltas, axis=1)
    score = 4.0 * corner_zone[:, 0] - 4.0 * corner_zone[:, 1] - 0.03 * dist_sq
    best_index = int(np.argmax(score))
    return corner_zone[best_index]


def snap_bottom_left_corner(point, contour, x_band=35.0, y_band=35.0, max_dist=60.0):
    """snap bottom left to points near the true bottom left corner area"""
    # work in float for distance math
    target = point.astype(np.float32)
    contour_float = contour.astype(np.float32)

    # bottom left corner should live near the min x and max y of the contour
    min_x = float(np.min(contour_float[:, 0]))
    max_y = float(np.max(contour_float[:, 1]))

    # first try the actual bottom left corner zone
    corner_zone = contour_float[
        (contour_float[:, 0] <= min_x + x_band) &
        (contour_float[:, 1] >= max_y - y_band)
    ]

    # if that is empty fall back to points near the guess
    if len(corner_zone) == 0:
        corner_zone = contour_float[
            (np.abs(contour_float[:, 0] - target[0]) <= max_dist) &
            (np.abs(contour_float[:, 1] - target[1]) <= max_dist)
        ]

    # if still empty just use the normal snap
    if len(corner_zone) == 0:
        return snap_point_to_contour(point, contour, max_dist=max_dist)

    # keep the candidate nearest to the guessed point
    deltas = corner_zone - target
    dist_sq = np.sum(deltas * deltas, axis=1)
    best_index = int(np.argmin(dist_sq))
    return corner_zone[best_index]


def snap_bottom_right_corner(point, contour, x_band=35.0, y_band=35.0, max_dist=60.0):
    """snap bottom right to points near the true bottom right corner area"""
    target = point.astype(np.float32)
    contour_float = contour.astype(np.float32)

    # bottom right corner should live near the max x and max y of the contour
    max_x = float(np.max(contour_float[:, 0]))
    max_y = float(np.max(contour_float[:, 1]))

    # first try the actual bottom right corner zone
    corner_zone = contour_float[
        (contour_float[:, 0] >= max_x - x_band) &
        (contour_float[:, 1] >= max_y - y_band)
    ]

    # if that is empty fall back to points near the guess
    if len(corner_zone) == 0:
        corner_zone = contour_float[
            (np.abs(contour_float[:, 0] - target[0]) <= max_dist) &
            (np.abs(contour_float[:, 1] - target[1]) <= max_dist)
        ]

    # if still empty just use the normal snap
    if len(corner_zone) == 0:
        return snap_point_to_contour(point, contour, max_dist=max_dist)

    # prefer the true outer corner, not the inside-biased line guess
    deltas = corner_zone - target
    dist_sq = np.sum(deltas * deltas, axis=1)
    score = 4.0 * corner_zone[:, 0] + 4.0 * corner_zone[:, 1] - 0.03 * dist_sq
    best_index = int(np.argmax(score))
    return corner_zone[best_index]


def find_bottom_right_turn_corner(point, contour, bottom_line, right_line, x_band=45.0, y_band=45.0, max_dist=80.0):
    """pick the contour turn point where the bottom run meets the right wall"""
    target = point.astype(np.float32)
    contour_float = contour.astype(np.float32)

    max_x = float(np.max(contour_float[:, 0]))
    max_y = float(np.max(contour_float[:, 1]))

    corner_zone = contour_float[
        (contour_float[:, 0] >= max_x - x_band) &
        (contour_float[:, 1] >= max_y - y_band)
    ]

    if len(corner_zone) == 0:
        corner_zone = contour_float[
            (np.abs(contour_float[:, 0] - target[0]) <= max_dist) &
            (np.abs(contour_float[:, 1] - target[1]) <= max_dist)
        ]

    if len(corner_zone) == 0:
        return snap_bottom_right_corner(point, contour, max_dist=max_dist)

    bottom_dist = np.array([point_line_distance(entry, bottom_line) for entry in corner_zone], dtype=np.float32)
    right_dist = np.array([point_line_distance(entry, right_line) for entry in corner_zone], dtype=np.float32)

    # keep the contour points that plausibly belong to the bottom/right transition
    bottom_limit = max(4.0, float(np.percentile(bottom_dist, 45)) * 1.4)
    right_limit = max(4.0, float(np.percentile(right_dist, 45)) * 1.4)
    turn_candidates = corner_zone[(bottom_dist <= bottom_limit) & (right_dist <= right_limit)]

    if len(turn_candidates) < 3:
        combined = bottom_dist + right_dist
        keep_count = min(len(corner_zone), max(3, len(corner_zone) // 5))
        best_indices = np.argsort(combined)[:keep_count]
        turn_candidates = corner_zone[best_indices]

    deltas = turn_candidates - target
    dist_sq = np.sum(deltas * deltas, axis=1)

    # among plausible turn points, prefer the one farthest out on the actual contour
    score = 2.5 * turn_candidates[:, 0] + 2.5 * turn_candidates[:, 1] - 0.04 * dist_sq
    best_index = int(np.argmax(score))
    return turn_candidates[best_index]
