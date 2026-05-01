from pathlib import Path
import json

import cv2
import numpy as np


DEFAULT_MARKER_DICT = "DICT_4X4_50"
DEFAULT_MARKER_LAYOUT = (
    {"id": 0, "center_uv": [0.08, 0.24], "size_uv": [0.22, 0.22]},
    {"id": 1, "center_uv": [0.50, 0.18], "size_uv": [0.22, 0.22]},
    {"id": 2, "center_uv": [0.92, 0.24], "size_uv": [0.22, 0.22]},
    {"id": 3, "center_uv": [0.08, 0.76], "size_uv": [0.22, 0.22]},
    {"id": 4, "center_uv": [0.50, 0.70], "size_uv": [0.22, 0.22]},
    {"id": 5, "center_uv": [0.92, 0.76], "size_uv": [0.22, 0.22]},
)
DEFAULT_MARKER_BOARD_MARGIN_FRACTION = 0.16


def ensure_aruco():
    """require OpenCV's aruco module for coded-board generation and detection"""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This workflow requires cv2.aruco. Install an OpenCV build with the aruco module available."
        )
    return cv2.aruco


def get_aruco_dictionary(dict_name=DEFAULT_MARKER_DICT):
    """load a named aruco dictionary from OpenCV"""
    aruco = ensure_aruco()
    dictionary_id = getattr(aruco, dict_name, None)
    if dictionary_id is None:
        raise RuntimeError(f"Unsupported ArUco dictionary: {dict_name}")
    return aruco.getPredefinedDictionary(dictionary_id)


def normalized_points_to_quad(points_uv, quad_xy):
    """map uv points in the unit square into an arbitrary quad"""
    quad_xy = np.array(quad_xy, dtype=np.float32).reshape(4, 2)
    homography = cv2.getPerspectiveTransform(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        quad_xy,
    )
    points_uv = np.array(points_uv, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_uv, homography)
    return transformed.reshape(-1, 2)


def inset_quad(quad_xy, inset_left=0.10, inset_top=0.10, inset_right=0.10, inset_bottom=0.10):
    """pull a quad inward by uv-style margins, preserving its projective shape"""
    corners_uv = np.array(
        [
            [float(inset_left), float(inset_top)],
            [1.0 - float(inset_right), float(inset_top)],
            [1.0 - float(inset_right), 1.0 - float(inset_bottom)],
            [float(inset_left), 1.0 - float(inset_bottom)],
        ],
        dtype=np.float32,
    )
    return normalized_points_to_quad(corners_uv, quad_xy)


def marker_patch_uv(center_uv, size_uv):
    """build one marker patch quad in board uv coordinates"""
    center_uv = np.array(center_uv, dtype=np.float32)
    size_uv = np.array(size_uv, dtype=np.float32)
    half = 0.5 * size_uv
    top_left = center_uv + np.array([-half[0], -half[1]], dtype=np.float32)
    top_right = center_uv + np.array([half[0], -half[1]], dtype=np.float32)
    bottom_right = center_uv + np.array([half[0], half[1]], dtype=np.float32)
    bottom_left = center_uv + np.array([-half[0], half[1]], dtype=np.float32)
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def marker_layout_with_projector_quads(board_quad_projector_px, marker_layout=None):
    """attach projector-space quads to each logical marker entry"""
    marker_layout = DEFAULT_MARKER_LAYOUT if marker_layout is None else marker_layout
    enriched = []
    for entry in marker_layout:
        patch_uv = marker_patch_uv(entry["center_uv"], entry["size_uv"])
        patch_projector = normalized_points_to_quad(patch_uv, board_quad_projector_px)
        enriched.append(
            {
                "id": int(entry["id"]),
                "center_uv": [float(entry["center_uv"][0]), float(entry["center_uv"][1])],
                "size_uv": [float(entry["size_uv"][0]), float(entry["size_uv"][1])],
                "projector_quad_px": patch_projector.astype(np.float32),
                "projector_center_px": np.mean(patch_projector, axis=0).astype(np.float32),
            }
        )
    return enriched


def square_board_rect_from_quad(
    board_quad_projector_px,
    canvas_width,
    canvas_height,
    pad_px=12,
    width_fraction=0.84,
    height_fraction=0.28,
    vertical_anchor="bottom",
    bottom_margin_fraction=0.10,
    use_bottom_band_fraction=0.52,
):
    """fit one axis-aligned board rectangle inside the projected target area"""
    quad = np.array(board_quad_projector_px, dtype=np.float32).reshape(4, 2)
    min_x = max(float(np.min(quad[:, 0])) + float(pad_px), 0.0)
    max_x = min(float(np.max(quad[:, 0])) - float(pad_px), float(canvas_width - 1))
    min_y = max(float(np.min(quad[:, 1])) + float(pad_px), 0.0)
    max_y = min(float(np.max(quad[:, 1])) - float(pad_px), float(canvas_height - 1))
    if max_x <= min_x or max_y <= min_y:
        raise RuntimeError("Projected board area collapsed while fitting a square marker rectangle")

    available_w = max_x - min_x
    available_h = max_y - min_y
    band_h = available_h * float(use_bottom_band_fraction)
    band_y0 = max_y - band_h
    bottom_margin = band_h * float(bottom_margin_fraction)
    band_y1 = max_y - bottom_margin
    if band_y1 <= band_y0:
        raise RuntimeError("Bottom-band board area collapsed while fitting a square marker rectangle")

    board_w = max(24.0, available_w * float(width_fraction))
    board_h = max(24.0, (band_y1 - band_y0) * float(height_fraction) / max(1e-6, float(use_bottom_band_fraction)))

    x0 = min_x + 0.5 * (available_w - board_w)
    if str(vertical_anchor).lower() == "bottom":
        y0 = band_y1 - board_h
    elif str(vertical_anchor).lower() == "center":
        y0 = band_y0 + 0.5 * ((band_y1 - band_y0) - board_h)
    else:
        y0 = band_y0

    x1 = x0 + board_w
    y1 = y0 + board_h
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def marker_layout_with_square_projector_quads(
    board_rect_projector_px,
    marker_layout=None,
    board_margin_fraction=DEFAULT_MARKER_BOARD_MARGIN_FRACTION,
):
    """attach square projector-space quads to each logical marker entry"""
    marker_layout = DEFAULT_MARKER_LAYOUT if marker_layout is None else marker_layout
    board_rect = np.array(board_rect_projector_px, dtype=np.float32).reshape(4)
    x0, y0, x1, y1 = [float(value) for value in board_rect]
    board_w = x1 - x0
    board_h = y1 - y0
    if board_w <= 1.0 or board_h <= 1.0:
        raise RuntimeError("Board rectangle is too small for square markers")

    margin_x = board_w * float(board_margin_fraction)
    margin_y = board_h * float(board_margin_fraction)
    inner_x0 = x0 + margin_x
    inner_y0 = y0 + margin_y
    inner_x1 = x1 - margin_x
    inner_y1 = y1 - margin_y
    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0
    if inner_w <= 1.0 or inner_h <= 1.0:
        raise RuntimeError("Board margin is too large for the remaining marker area")

    u_min = min(float(entry["center_uv"][0] - 0.5 * entry["size_uv"][0]) for entry in marker_layout)
    u_max = max(float(entry["center_uv"][0] + 0.5 * entry["size_uv"][0]) for entry in marker_layout)
    v_min = min(float(entry["center_uv"][1] - 0.5 * entry["size_uv"][1]) for entry in marker_layout)
    v_max = max(float(entry["center_uv"][1] + 0.5 * entry["size_uv"][1]) for entry in marker_layout)

    layout_w = max(1e-6, u_max - u_min)
    layout_h = max(1e-6, v_max - v_min)
    scale = min(inner_w / layout_w, inner_h / layout_h)

    used_w = layout_w * scale
    used_h = layout_h * scale
    x_origin = inner_x0 + 0.5 * (inner_w - used_w) - u_min * scale
    y_origin = inner_y0 + 0.5 * (inner_h - used_h) - v_min * scale

    enriched = []
    for entry in marker_layout:
        center_u = float(entry["center_uv"][0])
        center_v = float(entry["center_uv"][1])
        size_uv = float(entry["size_uv"][0])
        half_px = 0.5 * size_uv * scale
        center_x = x_origin + center_u * scale
        center_y = y_origin + center_v * scale
        top_left = [center_x - half_px, center_y - half_px]
        top_right = [center_x + half_px, center_y - half_px]
        bottom_right = [center_x + half_px, center_y + half_px]
        bottom_left = [center_x - half_px, center_y + half_px]
        projector_quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        enriched.append(
            {
                "id": int(entry["id"]),
                "center_uv": [center_u, center_v],
                "size_uv": [float(entry["size_uv"][0]), float(entry["size_uv"][1])],
                "projector_quad_px": projector_quad,
                "projector_center_px": np.array([center_x, center_y], dtype=np.float32),
            }
        )
    return enriched


def render_marker_pattern(
    canvas_shape,
    board_projector_spec,
    marker_layout=None,
    dict_name=DEFAULT_MARKER_DICT,
    marker_resolution_px=240,
    square_markers=True,
):
    """render a warped aruco marker board directly in projector coordinates"""
    dictionary = get_aruco_dictionary(dict_name)

    canvas_h, canvas_w = int(canvas_shape[0]), int(canvas_shape[1])
    preview = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    debug = preview.copy()

    if square_markers:
        board_rect = np.array(board_projector_spec, dtype=np.float32).reshape(4)
        x0, y0, x1, y1 = np.rint(board_rect).astype(np.int32).tolist()
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.rectangle(debug, (x0, y0), (x1, y1), (255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 200, 255), thickness=2, lineType=cv2.LINE_AA)
        marker_layout = marker_layout_with_square_projector_quads(board_rect, marker_layout)
    else:
        board_quad = np.rint(np.array(board_projector_spec, dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(preview, board_quad, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(debug, board_quad, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.polylines(debug, [board_quad.reshape(-1, 1, 2)], True, (0, 200, 255), 2, cv2.LINE_AA)
        marker_layout = marker_layout_with_projector_quads(board_projector_spec, marker_layout)

    source_square = np.array(
        [
            [0.0, 0.0],
            [marker_resolution_px - 1.0, 0.0],
            [marker_resolution_px - 1.0, marker_resolution_px - 1.0],
            [0.0, marker_resolution_px - 1.0],
        ],
        dtype=np.float32,
    )

    marker_specs = []
    for entry in marker_layout:
        marker_image = np.zeros((marker_resolution_px, marker_resolution_px), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, int(entry["id"]), marker_resolution_px, marker_image, 1)
        marker_bgr = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)

        destination = np.array(entry["projector_quad_px"], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(source_square, destination)

        warped = cv2.warpPerspective(
            marker_bgr,
            homography,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.warpPerspective(
            np.full((marker_resolution_px, marker_resolution_px), 255, dtype=np.uint8),
            homography,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        preview[warped_mask > 0] = warped[warped_mask > 0]
        debug[warped_mask > 0] = warped[warped_mask > 0]

        quad_int = np.rint(destination).astype(np.int32).reshape(-1, 1, 2)
        center = np.rint(entry["projector_center_px"]).astype(np.int32)
        cv2.polylines(debug, [quad_int], True, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(
            debug,
            str(int(entry["id"])),
            tuple(center + np.array([8, -8], dtype=np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )

        marker_specs.append(
            {
                "id": int(entry["id"]),
                "center_uv": entry["center_uv"],
                "size_uv": entry["size_uv"],
                "projector_center_px": entry["projector_center_px"].tolist(),
                "projector_quad_px": destination.tolist(),
            }
        )

    return {
        "preview": preview,
        "debug": debug,
        "board_projector_spec": np.array(board_projector_spec, dtype=np.float32).tolist(),
        "marker_dictionary": dict_name,
        "square_markers": bool(square_markers),
        "markers": marker_specs,
    }


def detect_marker_corners(image_bgr, dict_name=DEFAULT_MARKER_DICT):
    """detect marker corners and ids from a captured image"""
    dictionary = get_aruco_dictionary(dict_name)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) == 0:
        return []

    detections = []
    for corners_one, marker_id in zip(corners, ids.reshape(-1)):
        detections.append(
            {
                "id": int(marker_id),
                "image_quad_px": np.array(corners_one, dtype=np.float32).reshape(4, 2),
            }
        )
    return detections


def build_homography_correspondences(detections, marker_specs):
    """expand matched marker quads into corner correspondences"""
    marker_lookup = {int(spec["id"]): spec for spec in marker_specs}
    projector_points = []
    image_points = []
    matched_ids = []

    for detection in detections:
        marker_id = int(detection["id"])
        if marker_id not in marker_lookup:
            continue
        spec = marker_lookup[marker_id]
        projector_quad = np.array(spec["projector_quad_px"], dtype=np.float32).reshape(4, 2)
        image_quad = np.array(detection["image_quad_px"], dtype=np.float32).reshape(4, 2)
        projector_points.extend(projector_quad.tolist())
        image_points.extend(image_quad.tolist())
        matched_ids.append(marker_id)

    if len(projector_points) < 4:
        raise RuntimeError("Need at least one matched marker to build homography correspondences")

    return (
        np.array(projector_points, dtype=np.float32),
        np.array(image_points, dtype=np.float32),
        matched_ids,
    )


def solve_marker_homography(detections, marker_specs):
    """solve projector-to-camera and camera-to-projector homographies from matched markers"""
    projector_points, image_points, matched_ids = build_homography_correspondences(detections, marker_specs)
    homography_projector_to_camera, mask = cv2.findHomography(projector_points, image_points, cv2.RANSAC, 4.0)
    if homography_projector_to_camera is None:
        raise RuntimeError("cv2.findHomography failed for projector-to-camera solve")

    homography_camera_to_projector = np.linalg.inv(homography_projector_to_camera)
    if abs(float(homography_projector_to_camera[2, 2])) > 1e-9:
        homography_projector_to_camera = homography_projector_to_camera / float(homography_projector_to_camera[2, 2])
    if abs(float(homography_camera_to_projector[2, 2])) > 1e-9:
        homography_camera_to_projector = homography_camera_to_projector / float(homography_camera_to_projector[2, 2])

    projected = cv2.perspectiveTransform(projector_points.reshape(-1, 1, 2), homography_projector_to_camera).reshape(-1, 2)
    reprojection_errors = np.linalg.norm(projected - image_points, axis=1)

    return {
        "matched_marker_ids": [int(marker_id) for marker_id in matched_ids],
        "projector_points_px": projector_points,
        "camera_points_px": image_points,
        "homography_projector_to_camera": homography_projector_to_camera.astype(np.float64),
        "homography_camera_to_projector": homography_camera_to_projector.astype(np.float64),
        "inlier_mask": None if mask is None else mask.reshape(-1).astype(np.uint8),
        "reprojection_error_px": {
            "mean": float(np.mean(reprojection_errors)),
            "max": float(np.max(reprojection_errors)),
            "per_point": [float(value) for value in reprojection_errors],
        },
    }


def draw_detected_markers(image_bgr, detections, board_quad_camera_px=None):
    """draw detected marker ids and optional board quad for debugging"""
    debug = image_bgr.copy()
    if board_quad_camera_px is not None:
        quad = np.rint(np.array(board_quad_camera_px, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug, [quad], True, (0, 200, 255), 2, cv2.LINE_AA)

    for detection in detections:
        quad = np.rint(np.array(detection["image_quad_px"], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        center = np.rint(np.mean(detection["image_quad_px"], axis=0)).astype(np.int32)
        cv2.polylines(debug, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            debug,
            str(int(detection["id"])),
            tuple(center + np.array([8, -8], dtype=np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return debug


def save_json(data, output_path):
    """save calibration metadata as utf-8 json"""
    output_path = Path(output_path).resolve()
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path
