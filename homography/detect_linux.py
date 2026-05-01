from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import homography.charuco_projector_calib as charuco_base
import homography.final_projection_pipeline as base
from homography.version_model_config import MODEL_PATH

import logging
def _notify(progress, message: str) -> None:
    if progress is not None:
        progress.detail(message)
    logging.info(message)

# Syntax for sending progress updates
# 
# if progress is not None:
#        progress.step(n, "Increment overall steps.") 
# 
# _notify(progress, "Text update only.")

def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def projector_canvas_quad() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(base.PROJECTOR_WIDTH - 1), 0.0],
            [float(base.PROJECTOR_WIDTH - 1), float(base.PROJECTOR_HEIGHT - 1)],
            [0.0, float(base.PROJECTOR_HEIGHT - 1)],
        ],
        dtype=np.float64,
    )


def board_rect_quad(rect_xyxy: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = np.asarray(rect_xyxy, dtype=np.float64).reshape(4)
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.float64,
    )


def load_display_rect_from_meta(meta_path: Path) -> np.ndarray:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return np.asarray(meta["display_rect_xyxy"], dtype=np.float64)


def draw_quad(image_bgr: np.ndarray, quad_xy: np.ndarray, color: tuple[int, int, int], label: str | None = None) -> None:
    quad = np.rint(np.asarray(quad_xy, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image_bgr, [quad], True, color, 2, cv2.LINE_AA)
    if label:
        center = tuple(np.rint(np.mean(np.asarray(quad_xy, dtype=np.float64), axis=0)).astype(np.int32))
        cv2.putText(image_bgr, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_crosshair(image_bgr: np.ndarray, point_xy: np.ndarray, color: tuple[int, int, int], size: int = 10) -> None:
    x_val, y_val = np.rint(np.asarray(point_xy, dtype=np.float64)).astype(np.int32)
    cv2.line(image_bgr, (x_val - size, y_val), (x_val + size, y_val), color, 2, cv2.LINE_AA)
    cv2.line(image_bgr, (x_val, y_val - size), (x_val, y_val + size), color, 2, cv2.LINE_AA)
    cv2.circle(image_bgr, (x_val, y_val), max(3, size // 3), color, 2, cv2.LINE_AA)


def filter_charuco_corners_to_contour(
    corner_ids: np.ndarray,
    corners_px: np.ndarray,
    contour_xy: np.ndarray,
    *,
    min_signed_distance_px: float = -10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    contour = np.asarray(contour_xy, dtype=np.float32).reshape(-1, 1, 2)
    corner_ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1)
    corners_px = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    signed_distances = np.zeros(corner_ids.shape[0], dtype=np.float64)
    keep_mask = np.zeros(corner_ids.shape[0], dtype=bool)
    for index, point_xy in enumerate(corners_px):
        signed_distance = float(cv2.pointPolygonTest(contour, (float(point_xy[0]), float(point_xy[1])), True))
        signed_distances[index] = signed_distance
        keep_mask[index] = signed_distance >= float(min_signed_distance_px)
    return corner_ids[keep_mask], corners_px[keep_mask], keep_mask, signed_distances


def build_charuco_filter_overlay(
    captured_bgr: np.ndarray,
    contour_xy: np.ndarray,
    corner_ids: np.ndarray,
    corners_px: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    overlay = captured_bgr.copy()
    contour = np.rint(np.asarray(contour_xy, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], True, (255, 255, 0), 2, cv2.LINE_AA)
    for corner_id, point_xy, keep in zip(
        np.asarray(corner_ids, dtype=np.int32).reshape(-1),
        np.asarray(corners_px, dtype=np.float64).reshape(-1, 2),
        np.asarray(keep_mask, dtype=bool).reshape(-1),
    ):
        color = (0, 255, 0) if bool(keep) else (0, 0, 255)
        xy = tuple(np.rint(point_xy).astype(np.int32))
        cv2.circle(overlay, xy, 6, color, 2, cv2.LINE_AA)
        cv2.putText(overlay, str(int(corner_id)), (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    used_count = int(np.count_nonzero(keep_mask))
    total_count = int(np.asarray(keep_mask).size)
    cv2.putText(
        overlay,
        f"charuco kept {used_count}/{total_count} inside door contour",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def match_projector_camera_points_from_charuco_ids(
    projector_points_npz_path: Path,
    camera_ids: np.ndarray,
    camera_points_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(str(projector_points_npz_path))
    projector_ids = np.asarray(data["corner_ids"], dtype=np.int32).reshape(-1)
    projector_pts = np.asarray(data["projector_points_px"], dtype=np.float64).reshape(-1, 2)
    projector_by_id = {int(corner_id): projector_pts[index] for index, corner_id in enumerate(projector_ids)}
    camera_by_id = {
        int(corner_id): np.asarray(point_xy, dtype=np.float64)
        for corner_id, point_xy in zip(np.asarray(camera_ids, dtype=np.int32).reshape(-1), np.asarray(camera_points_px, dtype=np.float64).reshape(-1, 2))
    }
    matched_ids = sorted(set(projector_by_id.keys()) & set(camera_by_id.keys()))
    if len(matched_ids) < 4:
        raise RuntimeError(f"Only {len(matched_ids)} on-door ChArUco IDs were found; need at least 4.")
    projector_points_px = np.array([projector_by_id[corner_id] for corner_id in matched_ids], dtype=np.float64)
    camera_points_px = np.array([camera_by_id[corner_id] for corner_id in matched_ids], dtype=np.float64)
    return projector_points_px, camera_points_px, np.asarray(matched_ids, dtype=np.int32)


def compute_h_pc_reprojection_error(projector_points_px: np.ndarray, camera_points_px: np.ndarray, calib: base.CalibrationData, H_PC: np.ndarray) -> dict[str, object]:
    camera_ud_px = base.undistort_points_to_pinhole_pixels(camera_points_px, calib)
    projected = base.apply_homography(projector_points_px, H_PC)
    errors = np.linalg.norm(projected - camera_ud_px, axis=1)
    return {
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
        "per_point": [float(value) for value in errors],
    }


def build_scene_debug_overlay(
    scene_capture_bgr: np.ndarray,
    calib: base.CalibrationData,
    H_PC: np.ndarray,
    H_CD: np.ndarray,
    layout_base_door_xy_in: np.ndarray,
    bottom_unit: np.ndarray,
    up_unit: np.ndarray,
    bundle_data: dict[str, object],
    top_height_in: float,
    feature_result: dict[str, object],
    calibration_display_rect_xyxy: np.ndarray,
) -> np.ndarray:
    overlay = base.build_scene_layout_preview(
        scene_capture_bgr,
        calib,
        H_CD,
        layout_base_door_xy_in,
        bottom_unit,
        up_unit,
        bundle_data,
        top_height_in,
    )

    projector_canvas_camera = base.apply_homography(projector_canvas_quad(), H_PC)
    calibration_board_camera = base.apply_homography(board_rect_quad(calibration_display_rect_xyxy), H_PC)
    draw_quad(overlay, projector_canvas_camera, (255, 0, 255), "projector canvas")
    draw_quad(overlay, calibration_board_camera, (0, 165, 255), "charuco board")

    points = feature_result["points"]
    color_map = {
        "bottom_left": (255, 0, 0),
        "bottom_right": (0, 0, 255),
        "top_right": (255, 255, 0),
        "bottom_50": (0, 255, 255),
    }
    for name, color in color_map.items():
        point_ud = base.undistort_points_to_pinhole_pixels(np.asarray(points[name], dtype=np.float64).reshape(1, 2), calib)[0]
        draw_crosshair(overlay, point_ud, color, size=12)
        label_xy = tuple((np.rint(np.asarray(point_ud, dtype=np.float64)).astype(np.int32) + np.array([10, -10], dtype=np.int32)).tolist())
        cv2.putText(overlay, name, label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    layout_base_camera = base.door_points_to_camera_ud_px(np.asarray(layout_base_door_xy_in, dtype=np.float64).reshape(1, 2), H_CD)[0]
    draw_crosshair(overlay, layout_base_camera, (255, 255, 255), size=14)
    cv2.putText(
        overlay,
        "layout_base",
        tuple((np.rint(layout_base_camera).astype(np.int32) + np.array([10, -10], dtype=np.int32)).tolist()),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def build_projector_debug_canvas(
    projector_clean_bgr: np.ndarray,
    H_DP: np.ndarray,
    layout_base_door_xy_in: np.ndarray,
    bottom_unit: np.ndarray,
    up_unit: np.ndarray,
    bundle_data: dict[str, object],
    top_height_in: float,
) -> tuple[np.ndarray, dict[str, object]]:
    debug_canvas = projector_clean_bgr.copy()
    line_local = np.array([[0.0, 0.0], [0.0, float(top_height_in)]], dtype=np.float64)
    line_door = base.local_to_door_xy(line_local, layout_base_door_xy_in, bottom_unit, up_unit)
    line_projector = base.apply_homography(line_door, H_DP)
    cv2.line(
        debug_canvas,
        tuple(np.rint(line_projector[0]).astype(np.int32)),
        tuple(np.rint(line_projector[1]).astype(np.int32)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    draw_crosshair(debug_canvas, line_projector[0], (0, 255, 255), size=14)

    projector_decals = {}
    for name, quad_local in bundle_data["local_quads_uv_in"].items():
        quad_door = base.local_to_door_xy(quad_local, layout_base_door_xy_in, bottom_unit, up_unit)
        quad_projector = base.apply_homography(quad_door, H_DP)
        draw_quad(debug_canvas, quad_projector, (0, 255, 0), name)
        spec = base.get_decal_layout(base.DOOR_SIDE)[name]
        anchor_local_point = np.array([[float(spec["anchor_bottom_center_in"][0]), float(spec["anchor_bottom_center_in"][1])]], dtype=np.float64)
        anchor_door = base.local_to_door_xy(anchor_local_point, layout_base_door_xy_in, bottom_unit, up_unit)
        anchor_projector = base.apply_homography(anchor_door, H_DP)[0]
        draw_crosshair(debug_canvas, anchor_projector, (255, 0, 0), size=8)
        projector_decals[name] = {
            "quad_projector_px": quad_projector,
            "anchor_projector_px": anchor_projector,
        }

    return debug_canvas, projector_decals


def summarize_warnings(
    calibration_reprojection: dict[str, object],
    points: dict[str, np.ndarray],
    layout_base_door_xy_in: np.ndarray,
    bottom_unit: np.ndarray,
    up_unit: np.ndarray,
    projector_decals: dict[str, object],
    corner_to_intersection_layout_shift_in: float,
    charuco_used_count: int,
    charuco_rejected_count: int,
) -> list[str]:
    warnings: list[str] = []
    if float(calibration_reprojection["mean"]) > 3.0:
        warnings.append(
            f"Calibration reprojection mean is {float(calibration_reprojection['mean']):.2f}px; projector-to-camera solve may be weak."
        )
    if float(calibration_reprojection["max"]) > 8.0:
        warnings.append(
            f"Calibration reprojection max is {float(calibration_reprojection['max']):.2f}px; at least one calibration corner is far off."
        )
    bottom_width_camera = float(np.linalg.norm(np.asarray(points["bottom_right"], dtype=np.float64) - np.asarray(points["bottom_left"], dtype=np.float64)))
    if bottom_width_camera < 300.0:
        warnings.append(
            f"Detected bottom width in camera space is only {bottom_width_camera:.1f}px; detection may be noisy or the door is too small in frame."
        )
    orthogonality = abs(float(np.dot(base.normalize(bottom_unit), base.normalize(up_unit))))
    if orthogonality > 0.15:
        warnings.append(
            f"Derived door basis is not close to orthogonal (|dot|={orthogonality:.3f}); geometry derivation may be unstable."
        )
    for name, data in projector_decals.items():
        quad = np.asarray(data["quad_projector_px"], dtype=np.float64)
        min_x = float(np.min(quad[:, 0]))
        max_x = float(np.max(quad[:, 0]))
        min_y = float(np.min(quad[:, 1]))
        max_y = float(np.max(quad[:, 1]))
        if min_x < 0.0 or min_y < 0.0 or max_x > float(base.PROJECTOR_WIDTH - 1) or max_y > float(base.PROJECTOR_HEIGHT - 1):
            warnings.append(
                f"Projected decal '{name}' extends outside the 1920x1080 projector canvas."
            )
    if not np.isfinite(layout_base_door_xy_in).all():
        warnings.append("Layout base contains non-finite values; placement solve failed numerically.")
    if float(corner_to_intersection_layout_shift_in) > 0.25:
        warnings.append(
            f"Contour-corner layout anchor differs from the right-intersection anchor by {float(corner_to_intersection_layout_shift_in):.2f}in."
        )
    if int(charuco_used_count) < 12:
        warnings.append(
            f"Only {int(charuco_used_count)} on-door ChArUco corners were used; projector calibration may be underconstrained."
        )
    if int(charuco_rejected_count) > 0:
        warnings.append(
            f"Rejected {int(charuco_rejected_count)} ChArUco corners outside the detected door contour."
        )
    if not warnings:
        warnings.append("No obvious diagnostic threshold was exceeded in this run.")
    return warnings


def build_text_report(summary: dict[str, object]) -> str:
    lines = [
        "Detect And Project Debug Report",
        "",
        f"Door side: {summary['door_side']}",
        f"Anchor mode: {summary['anchor_mode']}",
        f"Calibration mode: {summary['charuco_mode']}",
        f"Calibration reprojection mean/max: {summary['calibration_reprojection_px']['mean']:.2f}px / {summary['calibration_reprojection_px']['max']:.2f}px",
        f"Matched ChArUco ids: {summary['matched_charuco_ids']}",
        f"ChArUco raw/on-door/rejected corners: {summary['charuco_raw_count']} / {summary['charuco_used_count']} / {summary['charuco_rejected_count']}",
        f"ChArUco display rect on projector: {summary['charuco_display_rect_xyxy']}",
        f"Contour-corner vs right-intersection layout shift: {summary['corner_to_intersection_layout_shift_in']:.2f}in",
        f"Layout base on door inches: {summary['layout_base_door_xy_in']}",
        f"Bottom unit: {summary['bottom_unit_door']}",
        f"Up unit: {summary['up_unit_door']}",
        "",
        "Detected camera points:",
    ]
    for name, point in summary["detected_camera_points_px"].items():
        lines.append(f"  {name}: {point}")
    lines.append("")
    lines.append("Warnings:")
    for warning in summary["warnings"]:
        lines.append(f"  - {warning}")
    lines.append("")
    lines.append("Saved files:")
    for name, path in summary["saved_files"].items():
        lines.append(f"  {name}: {path}")
    lines.append("")
    return "\n".join(lines)


def main(progress=None, argv=None) -> None:
    
    # Syntax for sending progress updates
    # 
    # if progress is not None:
    #        progress.step(n, "Increment overall steps.") 
    # 
    # _notify(progress, "Text update only.")
    
    _notify(progress, "Initializing pipeline.")
    parser = argparse.ArgumentParser(
        description="Run one detect-and-project pass and save a diagnostics bundle that explains where placement can go wrong."
    )
    parser.add_argument("--output-dir", default=str(base.OUTPUT_DIR / "debug_runs"), help="Directory for the diagnostics bundle.")
    parser.add_argument("--save-stem", default="debug", help="Base name used for saved artifacts.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the YOLO weights.")
    parser.add_argument("--door-side", default=base.DOOR_SIDE, choices=["left", "right"], help="Door variant to project.")
    parser.add_argument("--bundle-ppi", type=float, default=base.BUNDLE_PPI, help="Bundle raster PPI used for final warping.")
    args = parser.parse_args(argv if argv is not None else [])

    _notify(progress, "Loading neural network.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    base.DOOR_SIDE = args.door_side
    calib = base.CalibrationData(camera_matrix=base.CAMERA_MATRIX, dist_coeffs=base.DIST_COEFFS)
    charuco_dir = output_dir / f"{args.save_stem}_charuco_assets"
    charuco_base.CHARUCO_MODE_FULLSCREEN = True
    charuco_base.CHARUCO_MODE_ALIGN_BOX = False
    charuco_spec = charuco_base.CharucoSpec()
    charuco_image_path, charuco_meta_path, charuco_points_path, charuco_canvas = charuco_base.generate_charuco_projector_assets(charuco_dir, charuco_spec)
    calibration_display_rect_xyxy = load_display_rect_from_meta(charuco_meta_path)
    scene_canvas = np.full((base.PROJECTOR_HEIGHT, base.PROJECTOR_WIDTH, 3), base.SCENE_CANVAS_COLOR, dtype=np.uint8)

    try:
        calibration_capture = None
        scene_capture = None
        camera = base.open_camera()
        try:
            base.PROJECTOR_WINDOW.open()
            base.show_fullscreen(cv2.cvtColor(charuco_canvas, cv2.COLOR_GRAY2BGR))
            base.PROJECTOR_WINDOW.wait_for_settle()
            calibration_capture = base.rotate_frame(base.read_latest_frame(camera, base.CAPTURE_READS))
            if progress is not None:
                progress.step(1, "Captured ChArUco calibration.")

            base.show_fullscreen(scene_canvas)
            base.PROJECTOR_WINDOW.wait_for_settle()
            scene_capture = base.rotate_frame(base.read_latest_frame(camera, base.CAPTURE_READS))
            if progress is not None:
                progress.step(2, "Captured door features.")
        finally:
            camera.release()

        calibration_capture_path = output_dir / f"{args.save_stem}_calibration_capture.jpg"
        scene_capture_path = output_dir / f"{args.save_stem}_scene_capture.jpg"
        calibration_pattern_path = output_dir / f"{args.save_stem}_calibration_pattern.png"
        charuco_raw_detect_path = output_dir / f"{args.save_stem}_charuco_detected_raw.png"
        charuco_filter_path = output_dir / f"{args.save_stem}_charuco_detected_filtered.png"
        feature_overlay_path = output_dir / f"{args.save_stem}_feature_overlay.png"
        scene_debug_path = output_dir / f"{args.save_stem}_scene_debug_overlay.png"
        projector_clean_path = output_dir / f"{args.save_stem}_projector_clean.png"
        projector_debug_path = output_dir / f"{args.save_stem}_projector_debug.png"
        summary_json_path = output_dir / f"{args.save_stem}_summary.json"
        report_txt_path = output_dir / f"{args.save_stem}_report.txt"

        cv2.imwrite(str(calibration_pattern_path), cv2.cvtColor(charuco_canvas, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(calibration_capture_path), calibration_capture)
        cv2.imwrite(str(scene_capture_path), scene_capture)

        _notify(progress, "Running feature extraction.")
        feature_result = base.detect_door_feature_points(scene_capture_path, model_path)
        feature_overlay = base.build_debug_overlay(feature_result)
        cv2.imwrite(str(feature_overlay_path), feature_overlay)

        detected_ids_raw, detected_corners_raw = charuco_base.detect_charuco_corners_in_camera(calibration_capture, calib, charuco_meta_path)
        cv2.imwrite(str(charuco_raw_detect_path), charuco_base.draw_charuco_detection(calibration_capture, detected_ids_raw, detected_corners_raw))
        detected_ids, detected_corners, charuco_keep_mask, charuco_signed_distances = filter_charuco_corners_to_contour(
            detected_ids_raw,
            detected_corners_raw,
            feature_result["contour"],
        )
        
        if progress is not None:
            progress.step(3, "Calculated door pose.")

        cv2.imwrite(
            str(charuco_filter_path),
            build_charuco_filter_overlay(
                calibration_capture,
                feature_result["contour"],
                detected_ids_raw,
                detected_corners_raw,
                charuco_keep_mask,
            ),
        )

        if progress is not None:
            progress.step(4, "Applying homography.")
        projector_points_px, camera_points_px, matched_charuco_ids = match_projector_camera_points_from_charuco_ids(
            charuco_points_path,
            detected_ids,
            detected_corners,
        )
        charuco_raw_count = int(detected_ids_raw.shape[0])
        charuco_used_count = int(detected_ids.shape[0])
        charuco_rejected_count = int(charuco_raw_count - charuco_used_count)
        H_PC = base.compute_H_PC(projector_points_px, camera_points_px, calib)
        calibration_reprojection = compute_h_pc_reprojection_error(projector_points_px, camera_points_px, calib, H_PC)

        points = feature_result["points"]
        camera_feature_points = np.array([points[name] for name in base.DOOR_FEATURE_NAMES], dtype=np.float64)
        H_CD = base.compute_H_CD(camera_feature_points, base.DOOR_FEATURE_POINTS_XY_IN, calib)
        H_PD, H_DP = base.compose_H_PD(H_CD, H_PC)

        if progress is not None:
            progress.step(5, "Calculating decal locations.")
        layout_geometry = base.derive_layout_geometry_from_right_intersection(feature_result, H_CD, calib)
        layout_base_door_xy_in = np.asarray(layout_geometry["layout_base_door_xy_in"], dtype=np.float64)
        bottom_unit = np.asarray(layout_geometry["bottom_unit_door"], dtype=np.float64)
        up_unit = np.asarray(layout_geometry["up_unit_door"], dtype=np.float64)
        corner_layout_geometry = base.derive_layout_geometry_from_corners(points, H_CD, calib)
        corner_layout_base_door_xy_in = np.asarray(corner_layout_geometry["layout_base_door_xy_in"], dtype=np.float64)
        corner_to_intersection_layout_shift_in = float(
            np.linalg.norm(corner_layout_base_door_xy_in - layout_base_door_xy_in)
        )
        layout = base.get_decal_layout(base.DOOR_SIDE)
        bundle_data = base.build_layout_bundle(layout, float(args.bundle_ppi), base.BUNDLE_PADDING_IN)
        bundle_corners_door_xy_in = base.local_to_door_xy(
            bundle_data["bundle_corners_local_uv_in"],
            layout_base_door_xy_in,
            bottom_unit,
            up_unit,
        )
        projector_bundle_rgba, H_src_to_projector = base.warp_bundle_to_projector(bundle_data["image_rgba"], bundle_corners_door_xy_in, H_PD)
        projector_clean_bgr = base.flatten_rgba_over_black(projector_bundle_rgba)

        top_height_in = float(layout_geometry["top_height_in"])
        top_height_in = max(top_height_in, float(bundle_data["bounds_uv_in"][3]))
        scene_debug_overlay = build_scene_debug_overlay(
            feature_result["original"],
            calib,
            H_PC,
            H_CD,
            layout_base_door_xy_in,
            bottom_unit,
            up_unit,
            bundle_data,
            top_height_in,
            feature_result,
            calibration_display_rect_xyxy,
        )
        cv2.imwrite(str(scene_debug_path), scene_debug_overlay)

        if progress is not None:
            progress.step(6, "Compositing corrected decal layout.")

        projector_debug_bgr, projector_decals = build_projector_debug_canvas(
            projector_clean_bgr,
            H_DP,
            layout_base_door_xy_in,
            bottom_unit,
            up_unit,
            bundle_data,
            top_height_in,
        )
        cv2.imwrite(str(projector_clean_path), projector_clean_bgr)
        cv2.imwrite(str(projector_debug_path), projector_debug_bgr)

        warnings = summarize_warnings(
            calibration_reprojection,
            points,
            layout_base_door_xy_in,
            bottom_unit,
            up_unit,
            projector_decals,
            corner_to_intersection_layout_shift_in,
            charuco_used_count,
            charuco_rejected_count,
        )

        saved_files = {
            "charuco_pattern": calibration_pattern_path,
            "charuco_asset_image": charuco_image_path,
            "charuco_asset_meta": charuco_meta_path,
            "charuco_asset_points": charuco_points_path,
            "calibration_capture": calibration_capture_path,
            "charuco_detected_raw": charuco_raw_detect_path,
            "charuco_detected_filtered": charuco_filter_path,
            "scene_capture": scene_capture_path,
            "feature_overlay": feature_overlay_path,
            "scene_debug_overlay": scene_debug_path,
            "projector_clean": projector_clean_path,
            "projector_debug": projector_debug_path,
            "summary_json": summary_json_path,
            "report_txt": report_txt_path,
        }
        summary = {
            "door_side": base.DOOR_SIDE,
            "anchor_mode": str(layout_geometry.get("anchor_mode", "right_intersection")),
            "bundle_ppi": float(args.bundle_ppi),
            "charuco_mode": "fullscreen",
            "matched_charuco_ids": [int(marker_id) for marker_id in matched_charuco_ids],
            "charuco_raw_count": charuco_raw_count,
            "charuco_used_count": charuco_used_count,
            "charuco_rejected_count": charuco_rejected_count,
            "charuco_detected_ids_raw": [int(marker_id) for marker_id in detected_ids_raw],
            "charuco_detected_ids_used": [int(marker_id) for marker_id in detected_ids],
            "charuco_rejected_ids": [int(marker_id) for marker_id in detected_ids_raw[np.logical_not(charuco_keep_mask)]],
            "charuco_signed_distance_px": [float(value) for value in charuco_signed_distances],
            "calibration_reprojection_px": calibration_reprojection,
            "charuco_display_rect_xyxy": calibration_display_rect_xyxy,
            "detected_camera_points_px": {name: np.asarray(points[name], dtype=np.float64) for name in points},
            "bottom_left_seed_camera_px": np.asarray(feature_result["bottom_left_seed"], dtype=np.float64),
            "bottom_right_seed_camera_px": np.asarray(feature_result["bottom_right_seed"], dtype=np.float64),
            "top_right_seed_camera_px": np.asarray(feature_result["top_right_seed"], dtype=np.float64),
            "right_intersection_door_xy_in": np.asarray(layout_geometry["right_intersection_door_xy_in"], dtype=np.float64),
            "corner_layout_base_door_xy_in": corner_layout_base_door_xy_in,
            "corner_to_intersection_layout_shift_in": corner_to_intersection_layout_shift_in,
            "layout_base_camera_ud_px": np.asarray(layout_geometry["layout_base_camera_ud_px"], dtype=np.float64),
            "layout_base_door_xy_in": np.asarray(layout_base_door_xy_in, dtype=np.float64),
            "bottom_unit_door": np.asarray(bottom_unit, dtype=np.float64),
            "up_unit_door": np.asarray(up_unit, dtype=np.float64),
            "projector_decals": projector_decals,
            "H_PC": H_PC,
            "H_CD": H_CD,
            "H_PD": H_PD,
            "H_DP": H_DP,
            "H_src_to_projector": H_src_to_projector,
            "warnings": warnings,
            "saved_files": saved_files,
        }
        summary_json_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
        report_txt_path.write_text(build_text_report(json_ready(summary)), encoding="utf-8")

        print(f"calibration reprojection mean/max: {calibration_reprojection['mean']:.2f}px / {calibration_reprojection['max']:.2f}px")
        for warning in warnings:
            print(f"warning: {warning}")
        print(f"scene debug overlay: {scene_debug_path}")
        print(f"projector debug canvas: {projector_debug_path}")
        print(f"report: {report_txt_path}")

        base.show_fullscreen(projector_debug_bgr) # projector_(debug/clean)_bgr depending on desired detail
        print("Showing projector debug canvas fullscreen. Press any key in the projector window to close.")
    finally:
        if progress is not None:
            progress.step(7, "Displaying final output.")

if __name__ == "__main__":
    main()
