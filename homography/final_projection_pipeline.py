from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from homography.projector_display import ProjectorWindow, load_projector_config
from homography.door_feature_helper import build_debug_overlay, detect_door_feature_points
from homography.layout_spec import LAYOUT_LINE_FROM_RIGHT_IN, get_decal_layout
from homography.version_model_config import MODEL_PATH


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

DOOR_SIDE = "left"
WINDOW_NAME = "final projection pipeline"

CAMERA_INDEX = 0
CAMERA_ROTATE = "cw90"
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160
USE_DSHOW = True
CAPTURE_READS = 12
CAMERA_OPEN_TIMEOUT_SEC = 8.0
CAMERA_OPEN_RETRY_DELAY_SEC = 0.35
CAMERA_RELEASE_SETTLE_SEC = 0.5
DISPLAY_SETTLE_MS = 250
SHOW_PROJECTOR = True
SHOW_FINAL_PROJECTOR = True
SCENE_CANVAS_COLOR = (0, 0, 0)

PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
PROJECTOR_SIZE = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
MARKER_DICT = "DICT_4X4_50"
MARKER_BOARD_RECT = np.array([484.0, 700.0, 1328.0, 906.0], dtype=np.float32)
PROJECTOR_CONFIG = load_projector_config(PROJECTOR_WIDTH, PROJECTOR_HEIGHT, settle_ms=DISPLAY_SETTLE_MS)
PROJECTOR_WINDOW = ProjectorWindow(WINDOW_NAME, PROJECTOR_CONFIG)

BUNDLE_PPI = 80.0
BUNDLE_PADDING_IN = 0.75

CAMERA_MATRIX = np.array(
    [
        [3960.58477, 0.0, 1526.71088],
        [0.0, 3969.56508, 2140.11761],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST_COEFFS = np.array([0.11920817, -0.12795116, -0.00034711, 0.00135475, -0.25605914], dtype=np.float64)

# V4L2 may take a short moment to release /dev/video0 after a capture closes.
_CAMERA_LOCK = Lock()
_last_camera_release_at = 0.0

DOOR_FEATURE_NAMES = ("bottom_right", "bottom_left", "mirror_mount_right", "top_right")
DOOR_FEATURE_POINTS_XY_IN = np.array(
    [
        [0.0000, 0.0000],
        [-42.3970, -0.2300],
        [-32.0071, 25.9307],
        [0.2380, 62.2620],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


class ManagedCamera:
    def __init__(self, capture) -> None:
        self._capture = capture
        self._released = False

    def __getattr__(self, name: str):
        return getattr(self._capture, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def release(self) -> None:
        global _last_camera_release_at

        if self._released:
            return
        try:
            self._capture.release()
        finally:
            self._released = True
            _last_camera_release_at = time.monotonic()
            _CAMERA_LOCK.release()


def _wait_for_camera_release_settle() -> None:
    elapsed = time.monotonic() - _last_camera_release_at
    remaining = float(CAMERA_RELEASE_SETTLE_SEC) - elapsed
    if remaining > 0.0:
        time.sleep(remaining)


def open_camera(*, index: int | None = None, width: int | None = None, height: int | None = None):
    camera_index = int(CAMERA_INDEX if index is None else index)
    frame_width = int(CAMERA_WIDTH if width is None else width)
    frame_height = int(CAMERA_HEIGHT if height is None else height)
    backend = cv2.CAP_V4L2 if USE_DSHOW else 0

    _CAMERA_LOCK.acquire()
    try:
        _wait_for_camera_release_settle()
        deadline = time.monotonic() + float(CAMERA_OPEN_TIMEOUT_SEC)
        attempts = 0

        while True:
            attempts += 1
            camera = cv2.VideoCapture(camera_index, backend)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                return ManagedCamera(camera)

            camera.release()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Could not open camera index {camera_index} after {attempts} attempt(s) "
                    f"over {float(CAMERA_OPEN_TIMEOUT_SEC):.1f}s; another run or process may still be using it."
                )
            time.sleep(float(CAMERA_OPEN_RETRY_DELAY_SEC))
    except Exception:
        _CAMERA_LOCK.release()
        raise


def read_latest_frame(camera, reads: int = CAPTURE_READS) -> np.ndarray:
    frame = None
    for _ in range(int(reads)):
        ok, maybe_frame = camera.read()
        if ok and maybe_frame is not None and maybe_frame.size > 0:
            frame = maybe_frame
    if frame is None:
        raise RuntimeError("Could not read a valid camera frame.")
    return frame


def rotate_frame(frame: np.ndarray) -> np.ndarray:
    rotate_map = {
        "none": lambda image: image,
        "cw90": lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        "ccw90": lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "180": lambda image: cv2.rotate(image, cv2.ROTATE_180),
    }
    if CAMERA_ROTATE not in rotate_map:
        raise RuntimeError(f"Unsupported camera rotation: {CAMERA_ROTATE}")
    return rotate_map[CAMERA_ROTATE](frame)


def show_fullscreen(image: np.ndarray) -> None:
    PROJECTOR_WINDOW.show(image, label="final projection output")


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.array(vec, dtype=np.float64).reshape(-1)
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        raise RuntimeError("Cannot normalize a near-zero vector.")
    return vec / length


def as_points(points, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be Nx2. Got shape {arr.shape}.")
    return arr


def to_homogeneous(points_xy: np.ndarray) -> np.ndarray:
    return np.column_stack([points_xy, np.ones((points_xy.shape[0], 1), dtype=np.float64)])


def apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts_h = to_homogeneous(as_points(points_xy, "points_xy"))
    warped_h = (np.asarray(homography, dtype=np.float64) @ pts_h.T).T
    warped_h /= warped_h[:, [2]]
    return warped_h[:, :2]


def undistort_image(image_bgr: np.ndarray, calib: CalibrationData) -> np.ndarray:
    return cv2.undistort(image_bgr, calib.camera_matrix, calib.dist_coeffs, None, calib.camera_matrix)


def undistort_points_to_pinhole_pixels(points_px: np.ndarray, calib: CalibrationData) -> np.ndarray:
    points_px = as_points(points_px, "points_px")
    undistorted = cv2.undistortPoints(
        points_px.reshape(-1, 1, 2),
        calib.camera_matrix,
        calib.dist_coeffs,
        P=calib.camera_matrix,
    )
    return undistorted.reshape(-1, 2)


def compute_H_CD(camera_points_px: np.ndarray, door_points_xy_in: np.ndarray, calib: CalibrationData) -> np.ndarray:
    camera_points_px = as_points(camera_points_px, "camera_points_px")
    door_points_xy_in = as_points(door_points_xy_in, "door_points_xy_in")
    camera_ud_px = undistort_points_to_pinhole_pixels(camera_points_px, calib)
    K_inv = np.linalg.inv(calib.camera_matrix)
    camera_norm = (K_inv @ to_homogeneous(camera_ud_px).T).T[:, :2]
    H_norm_to_door, _ = cv2.findHomography(camera_norm, door_points_xy_in, cv2.RANSAC, 2.0)
    if H_norm_to_door is None:
        raise RuntimeError("Failed to solve camera-to-door homography.")
    H_CD = H_norm_to_door @ K_inv
    return H_CD / H_CD[2, 2]


def compute_H_PC(projector_points_px: np.ndarray, camera_points_px: np.ndarray, calib: CalibrationData) -> np.ndarray:
    projector_points_px = as_points(projector_points_px, "projector_points_px")
    camera_points_px = as_points(camera_points_px, "camera_points_px")
    camera_ud_px = undistort_points_to_pinhole_pixels(camera_points_px, calib)
    H_PC, _ = cv2.findHomography(projector_points_px, camera_ud_px, cv2.RANSAC, 2.0)
    if H_PC is None:
        raise RuntimeError("Failed to solve projector-to-camera homography.")
    return H_PC / H_PC[2, 2]


def compose_H_PD(H_CD: np.ndarray, H_PC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H_PD = np.asarray(H_CD, dtype=np.float64) @ np.asarray(H_PC, dtype=np.float64)
    H_PD /= H_PD[2, 2]
    H_DP = np.linalg.inv(H_PD)
    H_DP /= H_DP[2, 2]
    return H_PD, H_DP


def camera_points_to_door_xy(points_px: np.ndarray, H_CD: np.ndarray, calib: CalibrationData) -> np.ndarray:
    points_ud_px = undistort_points_to_pinhole_pixels(points_px, calib)
    return apply_homography(points_ud_px, H_CD)


def door_points_to_camera_ud_px(points_xy_in: np.ndarray, H_CD: np.ndarray) -> np.ndarray:
    H_DC = np.linalg.inv(np.asarray(H_CD, dtype=np.float64))
    H_DC /= H_DC[2, 2]
    return apply_homography(points_xy_in, H_DC)


def derive_layout_geometry_from_corners(
    points: dict[str, np.ndarray],
    H_CD: np.ndarray,
    calib: CalibrationData,
    *,
    layout_line_from_right_in: float = LAYOUT_LINE_FROM_RIGHT_IN,
) -> dict[str, np.ndarray | float]:
    camera_points_px = np.array(
        [
            points["bottom_right"],
            points["bottom_left"],
            points["top_right"],
        ],
        dtype=np.float64,
    )
    camera_points_ud_px = undistort_points_to_pinhole_pixels(camera_points_px, calib)
    door_points_xy_in = apply_homography(camera_points_ud_px, H_CD)
    bottom_right_door_xy_in, bottom_left_door_xy_in, top_right_door_xy_in = door_points_xy_in

    bottom_unit = normalize(bottom_right_door_xy_in - bottom_left_door_xy_in)
    up_raw = top_right_door_xy_in - bottom_right_door_xy_in
    up_unit = up_raw - bottom_unit * float(np.dot(up_raw, bottom_unit))
    up_unit = normalize(up_unit)
    if float(np.dot(up_unit, top_right_door_xy_in - bottom_right_door_xy_in)) < 0.0:
        up_unit = -up_unit

    bottom_width_in = float(np.linalg.norm(bottom_right_door_xy_in - bottom_left_door_xy_in))
    if bottom_width_in <= 1e-6:
        raise RuntimeError("Derived door bottom width collapsed while computing layout geometry.")
    layout_fraction = float(layout_line_from_right_in) / bottom_width_in
    layout_base_camera_ud_px = camera_points_ud_px[0] + (camera_points_ud_px[1] - camera_points_ud_px[0]) * layout_fraction
    layout_base_door_xy_in = apply_homography(layout_base_camera_ud_px.reshape(1, 2), H_CD)[0]
    top_height_in = float(np.dot(top_right_door_xy_in - layout_base_door_xy_in, up_unit))
    return {
        "bottom_right_camera_ud_px": camera_points_ud_px[0],
        "bottom_left_camera_ud_px": camera_points_ud_px[1],
        "top_right_camera_ud_px": camera_points_ud_px[2],
        "layout_base_camera_ud_px": layout_base_camera_ud_px,
        "bottom_right_door_xy_in": bottom_right_door_xy_in,
        "bottom_left_door_xy_in": bottom_left_door_xy_in,
        "top_right_door_xy_in": top_right_door_xy_in,
        "layout_base_door_xy_in": layout_base_door_xy_in,
        "bottom_unit_door": bottom_unit,
        "up_unit_door": up_unit,
        "top_height_in": top_height_in,
    }


def derive_layout_geometry_from_right_intersection(
    feature_result: dict[str, object],
    H_CD: np.ndarray,
    calib: CalibrationData,
    *,
    layout_line_from_right_in: float = LAYOUT_LINE_FROM_RIGHT_IN,
) -> dict[str, np.ndarray | float | str]:
    points = feature_result["points"]
    seed_points_px = np.array(
        [
            feature_result.get("bottom_right_seed", points["bottom_right"]),
            feature_result.get("bottom_left_seed", points["bottom_left"]),
            feature_result.get("top_right_seed", points["top_right"]),
        ],
        dtype=np.float64,
    )
    seed_points_ud_px = undistort_points_to_pinhole_pixels(seed_points_px, calib)
    seed_points_xy_in = apply_homography(seed_points_ud_px, H_CD)
    right_intersection_door_xy_in, bottom_left_seed_door_xy_in, top_right_seed_door_xy_in = seed_points_xy_in

    bottom_unit = normalize(right_intersection_door_xy_in - bottom_left_seed_door_xy_in)
    up_raw = top_right_seed_door_xy_in - right_intersection_door_xy_in
    up_unit = up_raw - bottom_unit * float(np.dot(up_raw, bottom_unit))
    up_unit = normalize(up_unit)
    if float(np.dot(up_unit, top_right_seed_door_xy_in - right_intersection_door_xy_in)) < 0.0:
        up_unit = -up_unit

    bottom_width_in = float(np.linalg.norm(right_intersection_door_xy_in - bottom_left_seed_door_xy_in))
    if bottom_width_in <= 1e-6:
        raise RuntimeError("Derived intersection bottom width collapsed while computing layout geometry.")
    layout_fraction = float(layout_line_from_right_in) / bottom_width_in
    layout_base_camera_ud_px = seed_points_ud_px[0] + (seed_points_ud_px[1] - seed_points_ud_px[0]) * layout_fraction
    layout_base_door_xy_in = apply_homography(layout_base_camera_ud_px.reshape(1, 2), H_CD)[0]
    top_height_in = float(np.dot(top_right_seed_door_xy_in - layout_base_door_xy_in, up_unit))
    return {
        "anchor_mode": "right_intersection",
        "right_intersection_camera_ud_px": seed_points_ud_px[0],
        "bottom_left_seed_camera_ud_px": seed_points_ud_px[1],
        "top_right_seed_camera_ud_px": seed_points_ud_px[2],
        "layout_base_camera_ud_px": layout_base_camera_ud_px,
        "right_intersection_door_xy_in": right_intersection_door_xy_in,
        "bottom_left_seed_door_xy_in": bottom_left_seed_door_xy_in,
        "top_right_seed_door_xy_in": top_right_seed_door_xy_in,
        "layout_base_door_xy_in": layout_base_door_xy_in,
        "bottom_unit_door": bottom_unit,
        "up_unit_door": up_unit,
        "top_height_in": top_height_in,
    }


def local_to_door_xy(points_uv_in: np.ndarray, layout_base_xy_in: np.ndarray, bottom_unit: np.ndarray, up_unit: np.ndarray) -> np.ndarray:
    points_uv_in = as_points(points_uv_in, "points_uv_in")
    layout_base_xy_in = np.asarray(layout_base_xy_in, dtype=np.float64).reshape(2)
    bottom_unit = normalize(bottom_unit)
    up_unit = normalize(up_unit)
    return (
        layout_base_xy_in.reshape(1, 2)
        + points_uv_in[:, [0]] * bottom_unit.reshape(1, 2)
        + points_uv_in[:, [1]] * up_unit.reshape(1, 2)
    )


def image_pixel_corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )


def load_rgba(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate([image, alpha], axis=2)
    if image.shape[2] == 4:
        return image
    raise RuntimeError(f"Unsupported image shape for {image_path}: {image.shape}")


def alpha_composite(base_rgba: np.ndarray, over_rgba: np.ndarray) -> np.ndarray:
    base = np.asarray(base_rgba, dtype=np.float32) / 255.0
    over = np.asarray(over_rgba, dtype=np.float32) / 255.0
    over_a = over[:, :, 3:4]
    base_a = base[:, :, 3:4]
    out_a = over_a + base_a * (1.0 - over_a)
    out_rgb_premult = over[:, :, :3] * over_a + base[:, :, :3] * base_a * (1.0 - over_a)
    out_rgb = np.zeros_like(out_rgb_premult)
    np.divide(out_rgb_premult, np.maximum(out_a, 1e-6), out=out_rgb, where=out_a > 1e-6)
    out = np.concatenate([out_rgb, out_a], axis=2)
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def flatten_rgba_over_black(image_rgba: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgba, dtype=np.float32) / 255.0
    alpha = image[:, :, 3:4]
    rgb = image[:, :, :3] * alpha
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def local_quad_from_anchor(anchor_u_in: float, anchor_v_in: float, width_in: float, height_in: float) -> np.ndarray:
    half_width = float(width_in) * 0.5
    bottom_left = np.array([anchor_u_in - half_width, anchor_v_in], dtype=np.float64)
    bottom_right = np.array([anchor_u_in + half_width, anchor_v_in], dtype=np.float64)
    top_left = bottom_left + np.array([0.0, float(height_in)], dtype=np.float64)
    top_right = bottom_right + np.array([0.0, float(height_in)], dtype=np.float64)
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)


def local_inches_to_bundle_pixels(points_uv_in: np.ndarray, bounds: tuple[float, float, float, float], ppi: float) -> np.ndarray:
    u_min, u_max, v_min, v_max = [float(value) for value in bounds]
    points_uv_in = as_points(points_uv_in, "points_uv_in")
    x_vals = (points_uv_in[:, 0] - u_min) * float(ppi)
    y_vals = (v_max - points_uv_in[:, 1]) * float(ppi)
    return np.column_stack([x_vals, y_vals]).astype(np.float64)


def build_layout_bundle(layout: dict[str, dict], ppi: float, padding_in: float) -> dict[str, object]:
    local_quads = {}
    all_corners = []
    for name, spec in layout.items():
        anchor_u, anchor_v = [float(value) for value in spec["anchor_bottom_center_in"]]
        width_in, height_in = [float(value) for value in spec["size_in"]]
        quad_local = local_quad_from_anchor(anchor_u, anchor_v, width_in, height_in)
        local_quads[name] = quad_local
        all_corners.append(quad_local)

    all_corners = np.vstack(all_corners)
    u_min = float(np.min(all_corners[:, 0]) - float(padding_in))
    u_max = float(np.max(all_corners[:, 0]) + float(padding_in))
    v_min = float(np.min(all_corners[:, 1]) - float(padding_in))
    v_max = float(np.max(all_corners[:, 1]) + float(padding_in))
    bounds = (u_min, u_max, v_min, v_max)

    canvas_width = int(np.ceil((u_max - u_min) * float(ppi))) + 1
    canvas_height = int(np.ceil((v_max - v_min) * float(ppi))) + 1
    bundle_rgba = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    for name, spec in layout.items():
        decal_rgba = load_rgba(Path(spec["image"]))
        target_quad_px = local_inches_to_bundle_pixels(local_quads[name], bounds, ppi)
        H_src_to_bundle = cv2.getPerspectiveTransform(
            image_pixel_corners(decal_rgba.shape[1], decal_rgba.shape[0]).astype(np.float32),
            target_quad_px.astype(np.float32),
        )
        warped = cv2.warpPerspective(
            decal_rgba,
            H_src_to_bundle,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        bundle_rgba = alpha_composite(bundle_rgba, warped)

    bundle_corners_local = np.array(
        [
            [u_min, v_max],
            [u_max, v_max],
            [u_max, v_min],
            [u_min, v_min],
        ],
        dtype=np.float64,
    )

    return {
        "image_rgba": bundle_rgba,
        "bounds_uv_in": bounds,
        "bundle_corners_local_uv_in": bundle_corners_local,
        "local_quads_uv_in": local_quads,
    }


def warp_bundle_to_projector(bundle_rgba: np.ndarray, bundle_corners_door_xy_in: np.ndarray, H_PD: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H_DP = np.linalg.inv(np.asarray(H_PD, dtype=np.float64))
    H_DP /= H_DP[2, 2]
    H_src_to_door = cv2.getPerspectiveTransform(
        image_pixel_corners(bundle_rgba.shape[1], bundle_rgba.shape[0]).astype(np.float32),
        as_points(bundle_corners_door_xy_in, "bundle_corners_door_xy_in").astype(np.float32),
    ).astype(np.float64)
    H_src_to_projector = H_DP @ H_src_to_door
    H_src_to_projector /= H_src_to_projector[2, 2]
    warped = cv2.warpPerspective(
        bundle_rgba,
        H_src_to_projector,
        PROJECTOR_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped, H_src_to_projector


def build_scene_layout_preview(
    scene_bgr: np.ndarray,
    calib: CalibrationData,
    H_CD: np.ndarray,
    layout_base_door_xy_in: np.ndarray,
    bottom_unit: np.ndarray,
    up_unit: np.ndarray,
    bundle_data: dict[str, object],
    layout_line_height_in: float,
) -> np.ndarray:
    preview = undistort_image(scene_bgr, calib)
    line_local = np.array([[0.0, 0.0], [0.0, float(layout_line_height_in)]], dtype=np.float64)
    line_door = local_to_door_xy(line_local, layout_base_door_xy_in, bottom_unit, up_unit)
    line_camera = door_points_to_camera_ud_px(line_door, H_CD)
    line_camera_int = np.rint(line_camera).astype(np.int32)
    cv2.line(preview, tuple(line_camera_int[0]), tuple(line_camera_int[1]), (0, 255, 255), 3, cv2.LINE_AA)

    for name, quad_local in bundle_data["local_quads_uv_in"].items():
        quad_door = local_to_door_xy(quad_local, layout_base_door_xy_in, bottom_unit, up_unit)
        quad_camera = door_points_to_camera_ud_px(quad_door, H_CD)
        quad_int = np.rint(quad_camera).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [quad_int], True, (0, 255, 0), 2, cv2.LINE_AA)
        label_point = tuple(np.rint(np.mean(quad_camera, axis=0)).astype(np.int32))
        cv2.putText(preview, name, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    return preview


def main() -> None:
    from homography.marker_calibration_utils import (
        build_homography_correspondences,
        detect_marker_corners,
        draw_detected_markers,
        render_marker_pattern,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calib = CalibrationData(camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS)

    marker_pattern = render_marker_pattern(
        (PROJECTOR_HEIGHT, PROJECTOR_WIDTH),
        MARKER_BOARD_RECT,
        dict_name=MARKER_DICT,
        square_markers=True,
    )
    scene_canvas = np.full((PROJECTOR_HEIGHT, PROJECTOR_WIDTH, 3), SCENE_CANVAS_COLOR, dtype=np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "final_marker_pattern.png"), marker_pattern["preview"])
    cv2.imwrite(str(OUTPUT_DIR / "final_marker_pattern_debug.png"), marker_pattern["debug"])

    if SHOW_PROJECTOR:
        PROJECTOR_WINDOW.open()

    camera = open_camera()
    try:
        if SHOW_PROJECTOR:
            show_fullscreen(marker_pattern["preview"])
            PROJECTOR_WINDOW.wait_for_settle()
        calibration_capture = rotate_frame(read_latest_frame(camera, CAPTURE_READS))

        if SHOW_PROJECTOR:
            show_fullscreen(scene_canvas)
            PROJECTOR_WINDOW.wait_for_settle()
        scene_capture = rotate_frame(read_latest_frame(camera, CAPTURE_READS))
    finally:
        camera.release()

    calibration_capture_path = OUTPUT_DIR / "final_calibration_capture.jpg"
    scene_capture_path = OUTPUT_DIR / "final_scene_capture.jpg"
    cv2.imwrite(str(calibration_capture_path), calibration_capture)
    cv2.imwrite(str(scene_capture_path), scene_capture)

    detections = detect_marker_corners(calibration_capture, MARKER_DICT)
    if not detections:
        raise RuntimeError("No ArUco markers detected in the calibration capture.")
    cv2.imwrite(
        str(OUTPUT_DIR / "final_calibration_detected_markers.png"),
        draw_detected_markers(calibration_capture, detections),
    )
    projector_points_px, camera_points_px, matched_marker_ids = build_homography_correspondences(
        detections,
        marker_pattern["markers"],
    )
    H_PC = compute_H_PC(projector_points_px, camera_points_px, calib)

    feature_result = detect_door_feature_points(scene_capture_path, MODEL_PATH)
    cv2.imwrite(str(OUTPUT_DIR / "final_scene_detected_features.png"), build_debug_overlay(feature_result))

    points = feature_result["points"]
    camera_feature_points = np.array([points[name] for name in DOOR_FEATURE_NAMES], dtype=np.float64)
    H_CD = compute_H_CD(camera_feature_points, DOOR_FEATURE_POINTS_XY_IN, calib)
    H_PD, H_DP = compose_H_PD(H_CD, H_PC)

    layout_geometry = derive_layout_geometry_from_right_intersection(feature_result, H_CD, calib)
    layout_base_door_xy_in = np.asarray(layout_geometry["layout_base_door_xy_in"], dtype=np.float64)
    bottom_unit = np.asarray(layout_geometry["bottom_unit_door"], dtype=np.float64)
    up_unit = np.asarray(layout_geometry["up_unit_door"], dtype=np.float64)
    layout = get_decal_layout(DOOR_SIDE)
    bundle_data = build_layout_bundle(layout, BUNDLE_PPI, BUNDLE_PADDING_IN)
    bundle_corners_door_xy_in = local_to_door_xy(
        bundle_data["bundle_corners_local_uv_in"],
        layout_base_door_xy_in,
        bottom_unit,
        up_unit,
    )
    projector_bundle_rgba, H_src_to_projector = warp_bundle_to_projector(bundle_data["image_rgba"], bundle_corners_door_xy_in, H_PD)

    top_height_in = float(layout_geometry["top_height_in"])
    top_height_in = max(top_height_in, float(bundle_data["bounds_uv_in"][3]))
    scene_preview = build_scene_layout_preview(
        feature_result["original"],
        calib,
        H_CD,
        layout_base_door_xy_in,
        bottom_unit,
        up_unit,
        bundle_data,
        top_height_in,
    )

    cv2.imwrite(str(OUTPUT_DIR / "final_scene_layout_preview.png"), scene_preview)
    cv2.imwrite(str(OUTPUT_DIR / "final_layout_bundle.png"), bundle_data["image_rgba"])
    cv2.imwrite(str(OUTPUT_DIR / "final_projector_bundle_rgba.png"), projector_bundle_rgba)
    final_projector_bgr = flatten_rgba_over_black(projector_bundle_rgba)
    cv2.imwrite(str(OUTPUT_DIR / "final_projector_bundle.png"), final_projector_bgr)

    if SHOW_PROJECTOR:
        if SHOW_FINAL_PROJECTOR:
            print("Displaying final warped projector image. Press any key in the projector window to close.")
            show_fullscreen(final_projector_bgr)
            cv2.waitKey(0)
        else:
            show_fullscreen(scene_canvas)
            cv2.waitKey(1)
        PROJECTOR_WINDOW.close()

    print(f"matched marker ids: {sorted(set(int(marker_id) for marker_id in matched_marker_ids))}")
    print(f"scene capture: {scene_capture_path}")
    print(f"calibration capture: {calibration_capture_path}")
    print(f"output projector image: {OUTPUT_DIR / 'final_projector_bundle.png'}")
    print(f"layout base on door inches: {layout_base_door_xy_in.tolist()}")
    print(f"bottom/right basis on door: {bottom_unit.tolist()} / {up_unit.tolist()}")
    print("saved:")
    print(OUTPUT_DIR / "final_marker_pattern.png")
    print(OUTPUT_DIR / "final_scene_detected_features.png")
    print(OUTPUT_DIR / "final_scene_layout_preview.png")
    print(OUTPUT_DIR / "final_layout_bundle.png")
    print(OUTPUT_DIR / "final_projector_bundle.png")
    _ = H_DP
    _ = H_src_to_projector


if __name__ == "__main__":
    main()
