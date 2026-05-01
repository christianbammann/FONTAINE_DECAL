from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECTOR_WIDTH = 1920
PROJECTOR_HEIGHT = 1080
ALIGN_CHARUCO_RECT = np.array([484.0, 700.0, 1328.0, 906.0], dtype=np.float64)

CHARUCO_MODE_FULLSCREEN = True
CHARUCO_MODE_ALIGN_BOX = False


@dataclass(frozen=True)
class CharucoSpec:
    dictionary_name: str = "DICT_5X5_1000"
    squares_x: int = 16
    squares_y: int = 9
    square_length_px: int = 120
    marker_length_px: int = 84
    projector_width: int = PROJECTOR_WIDTH
    projector_height: int = PROJECTOR_HEIGHT


def ensure_aruco() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This pipeline requires OpenCV with the aruco module available.")


def aruco_dictionary_from_name(name: str) -> cv2.aruco.Dictionary:
    ensure_aruco()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary name: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def build_charuco_board(spec: CharucoSpec) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(
        (int(spec.squares_x), int(spec.squares_y)),
        float(spec.square_length_px),
        float(spec.marker_length_px),
        aruco_dictionary_from_name(spec.dictionary_name),
    )


def validate_charuco_mode() -> None:
    enabled = int(bool(CHARUCO_MODE_FULLSCREEN)) + int(bool(CHARUCO_MODE_ALIGN_BOX))
    if enabled != 1:
        raise RuntimeError("Enable exactly one ChArUco projection mode.")


def _display_rect(spec: CharucoSpec) -> np.ndarray:
    validate_charuco_mode()
    if CHARUCO_MODE_FULLSCREEN:
        return np.array([0.0, 0.0, float(spec.projector_width), float(spec.projector_height)], dtype=np.float64)
    return np.asarray(ALIGN_CHARUCO_RECT, dtype=np.float64)


def _charuco_corner_points_px(board: cv2.aruco.CharucoBoard, origin_xy: np.ndarray) -> np.ndarray:
    return board.getChessboardCorners()[:, :2].astype(np.float64) + origin_xy.reshape(1, 2)


def generate_charuco_projector_assets(
    out_dir: str | Path,
    spec: CharucoSpec = CharucoSpec(),
) -> tuple[Path, Path, Path, np.ndarray]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    board = build_charuco_board(spec)
    display_rect = _display_rect(spec)
    x0, y0, x1, y1 = display_rect
    board_width = int(round(x1 - x0))
    board_height = int(round(y1 - y0))
    if board_width <= 0 or board_height <= 0:
        raise RuntimeError("Invalid ChArUco display rectangle.")

    board_img = board.generateImage((board_width, board_height), marginSize=0, borderBits=1)
    canvas = np.zeros((int(spec.projector_height), int(spec.projector_width)), dtype=np.uint8)
    canvas[int(round(y0)) : int(round(y1)), int(round(x0)) : int(round(x1))] = board_img

    corner_points_px = _charuco_corner_points_px(board, np.array([x0, y0], dtype=np.float64))
    corner_ids = np.arange(corner_points_px.shape[0], dtype=np.int32)

    image_path = out_dir / "charuco_projector.png"
    meta_path = out_dir / "charuco_projector_meta.json"
    points_path = out_dir / "charuco_projector_points.npz"

    cv2.imwrite(str(image_path), canvas)
    meta = asdict(spec)
    meta["display_rect_xyxy"] = display_rect.tolist()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(points_path, corner_ids=corner_ids, projector_points_px=corner_points_px)
    return image_path, meta_path, points_path, canvas


def detect_charuco_corners_in_camera(captured_bgr: np.ndarray, calib, charuco_meta_json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(Path(charuco_meta_json_path).read_text(encoding="utf-8"))
    board = build_charuco_board(
        CharucoSpec(
            dictionary_name=meta["dictionary_name"],
            squares_x=int(meta["squares_x"]),
            squares_y=int(meta["squares_y"]),
            square_length_px=int(meta["square_length_px"]),
            marker_length_px=int(meta["marker_length_px"]),
            projector_width=int(meta.get("projector_width", PROJECTOR_WIDTH)),
            projector_height=int(meta.get("projector_height", PROJECTOR_HEIGHT)),
        )
    )
    dictionary = aruco_dictionary_from_name(meta["dictionary_name"])

    gray = cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    else:
        detector_params = cv2.aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ArUco markers detected in the projected ChArUco capture.")

    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
        cameraMatrix=calib.camera_matrix,
        distCoeffs=calib.dist_coeffs,
    )
    if charuco_ids is None or num_corners is None or int(num_corners) < 4:
        raise RuntimeError("Not enough ChArUco corners detected.")
    return charuco_ids.reshape(-1).astype(np.int32), charuco_corners.reshape(-1, 2).astype(np.float64)


def build_projector_camera_calib_points_from_charuco(
    captured_bgr: np.ndarray,
    calib,
    charuco_meta_json_path: Path,
    projector_points_npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(str(projector_points_npz_path))
    projector_ids = np.asarray(data["corner_ids"], dtype=np.int32).reshape(-1)
    projector_pts = np.asarray(data["projector_points_px"], dtype=np.float64).reshape(-1, 2)
    camera_ids, camera_pts = detect_charuco_corners_in_camera(captured_bgr, calib, charuco_meta_json_path)

    projector_by_id = {int(corner_id): projector_pts[index] for index, corner_id in enumerate(projector_ids)}
    camera_by_id = {int(corner_id): camera_pts[index] for index, corner_id in enumerate(camera_ids)}
    matched_ids = sorted(set(projector_by_id.keys()) & set(camera_by_id.keys()))
    if len(matched_ids) < 4:
        raise RuntimeError(f"Only {len(matched_ids)} matched ChArUco IDs were found; need at least 4.")

    projector_points_px = np.array([projector_by_id[corner_id] for corner_id in matched_ids], dtype=np.float64)
    camera_points_px = np.array([camera_by_id[corner_id] for corner_id in matched_ids], dtype=np.float64)
    return projector_points_px, camera_points_px, np.asarray(matched_ids, dtype=np.int32)


def draw_charuco_detection(captured_bgr: np.ndarray, corner_ids: np.ndarray, corners_px: np.ndarray) -> np.ndarray:
    vis = captured_bgr.copy()
    for corner_id, point in zip(corner_ids.reshape(-1), corners_px.reshape(-1, 2)):
        xy = tuple(np.rint(point).astype(np.int32))
        cv2.circle(vis, xy, 6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, str(int(corner_id)), (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main() -> None:
    image_path, meta_path, points_path, _ = generate_charuco_projector_assets(Path(__file__).resolve().parent)
    print(f"Saved projector calibration image: {image_path}")
    print(f"Saved board metadata: {meta_path}")
    print(f"Saved projector corner map: {points_path}")


if __name__ == "__main__":
    main()
