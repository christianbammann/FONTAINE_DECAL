from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from homography.version_model_config import MODEL_PATH

SCRIPT_DIR = Path(__file__).resolve().parent
from homography.edge_line_utils import (
    detect_outer_right_line_from_image,
    fit_right_edge_from_contour,
    fit_right_edge_from_contour_midsection,
    refine_line,
    refine_right_line_by_global_offset,
    resnap_line_to_visible_edge,
)
from homography.line_math_utils import (
    extend_line_across_width,
    fit_line_from_points,
    find_bottom_right_turn_corner,
    intersect_lines,
    project_point_to_line,
    snap_bottom_left_corner,
    snap_top_right_corner,
)
from homography.mirror_point_utils import detect_mirror_points


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_IMAGE_PATH = SCRIPT_DIR / "Blank_Door_Photos" / "left_skew.jpg"

RIGHT_EDGE_INWARD_MARGIN_PX = 32
RIGHT_EDGE_OUTWARD_MARGIN_PX = 8
RIGHT_EDGE_ROI_BOTTOM_START_FRACTION = 0.10
RIGHT_EDGE_ROI_BOTTOM_END_FRACTION = 0.34

_MODEL = None
_MODEL_PATH = None


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _get_model(model_path: Path):
    global _MODEL, _MODEL_PATH
    model_path = str(Path(model_path).resolve())
    if _MODEL is None or _MODEL_PATH != model_path:
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


def clear_model_cache() -> None:
    """Clear the cached YOLO model to free GPU memory and prevent state issues."""
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        try:
            # Unload the model to release GPU memory
            _MODEL = None
            _MODEL_PATH = None
        except Exception:
            # Silently ignore any errors during cleanup
            pass


def load_segmented_door(image_path: Path, model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = _get_model(model_path).predict(str(Path(image_path).resolve()), verbose=False)
    if not results:
        raise RuntimeError("Model returned no results")
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise RuntimeError("No segmentation mask detected in the image")
    return result.orig_img.copy(), result.masks.data[0].cpu().numpy(), result.masks.xy[0].astype(np.int32)


def bottom_segment_endpoints(bottom_band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fitted = fit_line_from_points(bottom_band)
    direction = fitted[1] - fitted[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    if direction[0] < 0:
        direction = -direction
    offsets = bottom_band.astype(np.float32) - fitted[0]
    distances = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
    straight_points = bottom_band[distances <= max(5.0, float(np.percentile(distances, 35)) * 1.75)]
    if len(straight_points) < 6:
        straight_points = bottom_band
    along = np.dot(straight_points.astype(np.float32) - fitted[0], direction)
    return straight_points[int(np.argmin(along))].astype(np.int32), straight_points[int(np.argmax(along))].astype(np.int32)


def extract_points(contour: np.ndarray) -> dict[str, np.ndarray]:
    bottom_y = float(np.percentile(contour[:, 1], 99))
    bottom_band = contour[contour[:, 1] > bottom_y - 40.0]
    if len(bottom_band) < 2:
        raise RuntimeError("Could not find enough points along the bottom of the door")
    bottom_left, bottom_right = bottom_segment_endpoints(bottom_band)
    top_y = int(np.min(contour[:, 1]))
    top_band = contour[contour[:, 1] < top_y + 20]
    if len(top_band) == 0:
        raise RuntimeError("Could not find enough points along the top of the door")
    return {
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
        "bottom_50": np.rint((bottom_left + bottom_right) / 2.0).astype(np.int32),
        "top_right": top_band[np.argmax(top_band[:, 0])].astype(np.int32),
    }


def fit_bottom_line_from_contour_band(contour: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    contour_float = np.array(contour, dtype=np.float32)
    max_y = float(np.max(contour_float[:, 1]))
    min_x = float(np.min(contour_float[:, 0]))
    max_x = float(np.max(contour_float[:, 0]))
    band = contour_float[contour_float[:, 1] >= max_y - 34.0]
    if len(band) < 12:
        raise RuntimeError("Not enough contour points in bottom band")
    bucket_ids = np.floor((band[:, 0] - np.min(band[:, 0])) / 12.0).astype(np.int32)
    sampled = np.array([band[bucket_ids == bucket_id][np.argmax(band[bucket_ids == bucket_id][:, 1])] for bucket_id in np.unique(bucket_ids)], dtype=np.float32)
    trimmed = sampled[(sampled[:, 0] >= min_x + 0.08 * (max_x - min_x)) & (sampled[:, 0] <= max_x - 0.10 * (max_x - min_x))]
    if len(trimmed) < 8:
        raise RuntimeError("Bottom band became too sparse after trimming corners")
    fitted = fit_line_from_points(trimmed)
    direction = fitted[1] - fitted[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    distances = np.abs((trimmed - fitted[0])[:, 0] * direction[1] - (trimmed - fitted[0])[:, 1] * direction[0])
    inliers = trimmed[distances <= max(2.5, float(np.percentile(distances, 70)) * 1.4)]
    if len(inliers) < 6:
        inliers = trimmed
    roi = (
        int(np.min(trimmed[:, 0]) - 28),
        int(max_y - 34.0 - 18),
        int(np.max(trimmed[:, 0]) + 16),
        int(max_y + 2.0 + 8),
    )
    return fit_line_from_points(inliers), roi


def _format_v2_camera_feature_points_block(array: np.ndarray) -> str:
    points = np.array(array, dtype=np.float64).reshape(3, 2)
    labels = (
        "feature 0 (bottom right)",
        "feature 1 (bottom left)",
        "feature 2 (mirror right)",
    )
    lines = [
        "CAMERA_FEATURE_POINTS_2D = np.array(",
        "    [",
    ]
    for point, label in zip(points, labels):
        lines.append(f"        [{point[0]:.1f}, {point[1]:.1f}],  # {label}")
    lines.extend(
        [
            "    ],",
            "    dtype=np.float64,",
            ")",
        ]
    )
    return "\n".join(lines)


def _draw_full_line(image: np.ndarray, line: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    p1, p2 = np.rint(np.array(line, dtype=np.float32)).astype(np.int32)
    cv2.line(image, tuple(p1), tuple(p2), color, int(thickness), cv2.LINE_AA)


def _draw_point(image: np.ndarray, point: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    xy = np.rint(np.array(point, dtype=np.float32)).astype(np.int32)
    cv2.circle(image, tuple(xy), 7, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        tuple((xy + np.array([10, -10], dtype=np.int32)).tolist()),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _build_masks(mask: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape[:2]
    mask_uint8 = cv2.resize((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    boundary_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    mask_inside = cv2.erode(mask_uint8, np.ones((9, 9), np.uint8), iterations=1)
    return boundary_mask, mask_inside


def _build_bottom_roi(points: dict[str, np.ndarray], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    return (
        max(0, int(points["bottom_left"][0]) - 20),
        max(0, int(min(points["bottom_left"][1], points["bottom_right"][1])) - 45),
        min(width, int(points["bottom_right"][0]) + 20),
        min(height, int(max(points["bottom_left"][1], points["bottom_right"][1])) + 45),
    )


def _build_right_roi(
    points: dict[str, np.ndarray],
    mask_inside: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    top_right = np.array(points["top_right"], dtype=np.float32)
    bottom_right = np.array(points["bottom_right"], dtype=np.float32)
    y_top = float(min(top_right[1], bottom_right[1]))
    y_bottom = float(max(top_right[1], bottom_right[1]))
    span = max(1.0, y_bottom - y_top)

    band_y0 = max(0, int(np.floor(y_bottom - RIGHT_EDGE_ROI_BOTTOM_END_FRACTION * span)))
    band_y1 = min(height, int(np.ceil(y_bottom - RIGHT_EDGE_ROI_BOTTOM_START_FRACTION * span)))
    if band_y1 <= band_y0 + 12:
        band_y0 = max(0, int(np.floor(y_bottom - 0.40 * span)))
        band_y1 = min(height, int(np.ceil(y_bottom - 0.08 * span)))

    band_mask = mask_inside[band_y0:band_y1]
    row_maxima = []
    for local_row in range(band_mask.shape[0]):
        cols = np.where(band_mask[local_row] > 0)[0]
        if len(cols) == 0:
            continue
        row_maxima.append(float(np.max(cols)))

    if row_maxima:
        right_edge_x_hint = float(np.percentile(np.array(row_maxima, dtype=np.float32), 90.0))
    else:
        right_edge_x_hint = float(max(top_right[0], bottom_right[0]))

    right_edge_x_hint = float(
        np.clip(
            right_edge_x_hint,
            max(float(top_right[0]), float(bottom_right[0])) - 32.0,
            float(width - 1),
        )
    )

    focused_roi = (
        max(0, int(np.floor(right_edge_x_hint - RIGHT_EDGE_INWARD_MARGIN_PX))),
        band_y0,
        min(width, int(np.ceil(right_edge_x_hint + RIGHT_EDGE_OUTWARD_MARGIN_PX))),
        band_y1,
    )
    return top_right, bottom_right, right_edge_x_hint, focused_roi


def detect_door_feature_points(image_path: Path, model_path: Path | None = None) -> dict[str, object]:
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path).resolve()
    image_path = Path(image_path).resolve()

    original, mask, contour = load_segmented_door(image_path, model_path)
    rough_points = extract_points(contour)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    boundary_mask, mask_inside = _build_masks(mask, gray.shape)

    bottom_roi = _build_bottom_roi(rough_points, gray.shape)
    right_top_hint, right_bottom_hint, right_edge_x_hint, right_roi = _build_right_roi(rough_points, mask_inside, gray.shape)

    try:
        refined_bottom, _ = fit_bottom_line_from_contour_band(contour)
    except RuntimeError:
        refined_bottom = refine_line(
            gray,
            boundary_mask,
            mask_inside,
            bottom_roi,
            "horizontal",
            (rough_points["bottom_left"] + rough_points["bottom_right"]) / 2.0,
        )
    try:
        refined_bottom = resnap_line_to_visible_edge(
            gray,
            bottom_roi,
            refined_bottom,
            "horizontal",
            prefer_direction=-1,
        )
    except RuntimeError:
        pass
    refined_bottom = extend_line_across_width(refined_bottom, gray.shape[1])

    right_expected = np.array(
        [
            float(right_edge_x_hint),
            0.5 * float(right_roi[1] + right_roi[3]),
        ],
        dtype=np.float32,
    )
    try:
        refined_right = fit_right_edge_from_contour_midsection(
            contour,
            right_roi,
            np.rint(right_top_hint).astype(np.int32),
            np.rint(right_bottom_hint).astype(np.int32),
        )
        try:
            refined_right = refine_right_line_by_global_offset(
                gray,
                boundary_mask,
                right_roi,
                refined_right,
                inward_search_px=10,
                outward_search_px=6,
                max_inward_shift=8.0,
                max_outward_shift=6.0,
            )
        except RuntimeError:
            pass
    except RuntimeError:
        try:
            refined_right = fit_right_edge_from_contour(contour, right_roi, right_expected)
            try:
                refined_right = refine_right_line_by_global_offset(
                    gray,
                    boundary_mask,
                    right_roi,
                    refined_right,
                    inward_search_px=10,
                    outward_search_px=6,
                    max_inward_shift=8.0,
                    max_outward_shift=6.0,
                )
            except RuntimeError:
                pass
        except RuntimeError:
            try:
                refined_right = detect_outer_right_line_from_image(
                    gray,
                    boundary_mask,
                    right_roi,
                    right_expected,
                    inward_tolerance=10,
                    outward_tolerance=8,
                )
            except RuntimeError:
                refined_right = refine_line(
                    gray,
                    boundary_mask,
                    mask_inside,
                    right_roi,
                    "vertical",
                    right_expected,
                )
    try:
        refined_right = resnap_line_to_visible_edge(
            gray,
            right_roi,
            refined_right,
            "vertical",
            prefer_direction=-1,
            search_radius=8,
        )
    except RuntimeError:
        pass

    bottom_left_seed = project_point_to_line(rough_points["bottom_left"], refined_bottom)
    bottom_right_seed = intersect_lines(refined_bottom, refined_right)
    top_right_seed = project_point_to_line(right_top_hint, refined_right)

    bottom_left_corner = snap_bottom_left_corner(bottom_left_seed, contour)
    bottom_right_corner = find_bottom_right_turn_corner(
        bottom_right_seed,
        contour,
        refined_bottom,
        refined_right,
    )
    top_right_corner = snap_top_right_corner(top_right_seed, contour)

    points = {
        "bottom_left": np.rint(bottom_left_corner).astype(np.int32),
        "bottom_right": np.rint(bottom_right_corner).astype(np.int32),
        "top_right": np.rint(top_right_corner).astype(np.int32),
        "bottom_50": np.rint((bottom_left_corner + bottom_right_corner) / 2.0).astype(np.int32),
        "bottom_edge_line": np.array(refined_bottom, dtype=np.float32),
        "right_edge_line": np.array(refined_right, dtype=np.float32),
    }
    points.update(detect_mirror_points(gray, contour))

    v2_feature_points = None
    if "mirror_mount_right" in points:
        v2_feature_points = np.array(
            [
                points["bottom_right"],
                points["bottom_left"],
                points["mirror_mount_right"],
            ],
            dtype=np.float64,
        )

    return {
        "image_path": Path(image_path).resolve(),
        "model_path": model_path,
        "original": original,
        "mask": mask,
        "contour": contour,
        "rough_points": rough_points,
        "points": points,
        "bottom_roi": bottom_roi,
        "right_roi": right_roi,
        "bottom_left_seed": np.array(bottom_left_seed, dtype=np.float32),
        "bottom_right_seed": np.array(bottom_right_seed, dtype=np.float32),
        "top_right_seed": np.array(top_right_seed, dtype=np.float32),
        "v2_camera_feature_points_2d": v2_feature_points,
    }


def build_debug_overlay(result: dict[str, object]) -> np.ndarray:
    image = np.array(result["original"], copy=True)
    contour = np.rint(np.array(result["contour"], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(image, [contour], -1, (0, 255, 255), 1, cv2.LINE_AA)

    _draw_full_line(image, result["points"]["bottom_edge_line"], (0, 255, 0), 2)
    _draw_full_line(image, result["points"]["right_edge_line"], (255, 255, 0), 2)

    _draw_point(image, result["points"]["bottom_left"], "bottom_left", (255, 0, 0))
    _draw_point(image, result["points"]["bottom_right"], "bottom_right", (0, 0, 255))
    _draw_point(image, result["points"]["top_right"], "top_right", (255, 255, 0))

    if "mirror_mount_right" in result["points"]:
        _draw_point(image, result["points"]["mirror_mount_right"], "mirror_right", (255, 0, 255))
    if "mirror_mount_left" in result["points"]:
        _draw_point(image, result["points"]["mirror_mount_left"], "mirror_left", (200, 200, 255))

    return image


def save_outputs(result: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(result["image_path"])
    stem = image_path.stem

    overlay_path = output_dir / f"{stem}_door_feature_helper_overlay.png"
    json_path = output_dir / f"{stem}_door_feature_helper_points.json"

    overlay = build_debug_overlay(result)
    cv2.imwrite(str(overlay_path), overlay)

    payload = {
        "image_path": result["image_path"],
        "model_path": result["model_path"],
        "rough_points": result["rough_points"],
        "points": result["points"],
        "bottom_left_seed": result["bottom_left_seed"],
        "bottom_right_seed": result["bottom_right_seed"],
        "top_right_seed": result["top_right_seed"],
        "bottom_roi": result["bottom_roi"],
        "right_roi": result["right_roi"],
        "v2_camera_feature_points_2d": result["v2_camera_feature_points_2d"],
        "v2_camera_feature_point_order": ["bottom_right", "bottom_left", "mirror_mount_right"],
        "v2_camera_feature_points_2d_snippet": None
        if result["v2_camera_feature_points_2d"] is None
        else _format_v2_camera_feature_points_block(result["v2_camera_feature_points_2d"]),
        "note": "Bottom corners are snapped to contour corner points, not taken as the intersection of fitted edge lines.",
    }
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return overlay_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect bottom-left and bottom-right door corner points, plus mirror-right when available, for v2_plane_detector input prep."
    )
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE_PATH), help="Path to the door image")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the YOLO segmentation weights")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for debug overlay and JSON output")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    result = detect_door_feature_points(image_path, model_path)
    overlay_path, json_path = save_outputs(result, output_dir)

    points = result["points"]
    print(f"overlay saved to: {overlay_path}")
    print(f"points json saved to: {json_path}")
    print(f"bottom_left: {points['bottom_left'].tolist()}")
    print(f"bottom_right: {points['bottom_right'].tolist()}")
    if "mirror_mount_right" in points:
        print(f"mirror_mount_right: {points['mirror_mount_right'].tolist()}")
    else:
        print("mirror_mount_right: not detected")

    if result["v2_camera_feature_points_2d"] is not None:
        print("\nCAMERA_FEATURE_POINTS_2D for v2_plane_detector.py:")
        print(_format_v2_camera_feature_points_block(result["v2_camera_feature_points_2d"]))
    else:
        print("\nCAMERA_FEATURE_POINTS_2D block not emitted because mirror_mount_right was not detected.")


if __name__ == "__main__":
    main()
