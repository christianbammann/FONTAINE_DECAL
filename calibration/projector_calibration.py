from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_IMAGE = SCRIPT_DIR / "capture_checkerboard.jpg"
DEFAULT_PROJECTOR_IMAGE = SCRIPT_DIR / "checkerboard.png"
DEFAULT_H_PATH = SCRIPT_DIR / "H_ctp.npy"
DEFAULT_PATTERN_SIZE = (21, 10)

# Optional cache: set by get_h_ctp() so other modules can import quickly.
H_ctp: np.ndarray | None = None


def compute_h_ctp(
    camera_image_path: Path = DEFAULT_CAMERA_IMAGE,
    projector_image_path: Path = DEFAULT_PROJECTOR_IMAGE,
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
    show_preview: bool = False,
) -> np.ndarray:
    """Compute camera->projector homography from checkerboard images."""
    img = cv2.imread(str(camera_image_path))
    img_p = cv2.imread(str(projector_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read camera checkerboard image: {camera_image_path}")
    if img_p is None:
        raise FileNotFoundError(f"Could not read projector checkerboard image: {projector_image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    ret_p, corners_p = cv2.findChessboardCorners(gray_p, pattern_size)
    if not ret or not ret_p:
        raise RuntimeError("Chessboard corners were not detected in one or both images.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners_p = cv2.cornerSubPix(gray_p, corners_p, (11, 11), (-1, -1), criteria)

    if show_preview:
        prev_cam = img.copy()
        prev_proj = img_p.copy()
        cv2.drawChessboardCorners(prev_cam, pattern_size, corners, ret)
        cv2.drawChessboardCorners(prev_proj, pattern_size, corners_p, ret_p)
        cv2.imshow("Camera corners", prev_cam)
        cv2.imshow("Projector corners", prev_proj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    camera_points = np.squeeze(corners).astype(np.float32)
    projector_points = np.squeeze(corners_p).astype(np.float32)
    h_ctp, _ = cv2.findHomography(camera_points, projector_points)
    if h_ctp is None:
        raise RuntimeError("cv2.findHomography failed to return a matrix.")
    return h_ctp


def save_h_ctp(h_ctp: np.ndarray, output_path: Path = DEFAULT_H_PATH) -> Path:
    output_path = Path(output_path)
    np.save(str(output_path), h_ctp)
    return output_path


def load_h_ctp(path: Path = DEFAULT_H_PATH) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Homography file not found: {path}")
    h_ctp = np.load(str(path))
    if h_ctp.shape != (3, 3):
        raise ValueError(f"Expected (3,3) homography matrix, got {h_ctp.shape}")
    return h_ctp.astype(np.float64)


def get_h_ctp(prefer_saved: bool = True, save_if_computed: bool = True) -> np.ndarray:
    """
    Return camera->projector homography, preferring cached/saved matrix.
    This is safe to call from other modules (no preview windows by default).
    """
    global H_ctp

    if H_ctp is not None:
        return H_ctp.copy()

    if prefer_saved and DEFAULT_H_PATH.exists():
        H_ctp = load_h_ctp(DEFAULT_H_PATH)
        return H_ctp.copy()

    H_ctp = compute_h_ctp(show_preview=False).astype(np.float64)
    if save_if_computed:
        save_h_ctp(H_ctp, DEFAULT_H_PATH)
    return H_ctp.copy()


def _demo():
    h_ctp = get_h_ctp(prefer_saved=False, save_if_computed=True)
    print("H_ctp:")
    print(h_ctp)

    pt_cam = np.array([[[1357.0, 3169.0]]], dtype=np.float32)
    pt_proj = cv2.perspectiveTransform(pt_cam, h_ctp.astype(np.float32))
    x_p, y_p = pt_proj[0][0]
    print(f"Mapped test point -> projector: ({x_p:.3f}, {y_p:.3f})")


if __name__ == "__main__":
    _demo()
