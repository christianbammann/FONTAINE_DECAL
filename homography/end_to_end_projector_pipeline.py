from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import homography.charuco_projector_calib as _charuco
from homography.charuco_projector_calib import (
    ALIGN_CHARUCO_RECT,
    PROJECTOR_HEIGHT,
    PROJECTOR_WIDTH,
    CharucoSpec,
    aruco_dictionary_from_name,
    build_charuco_board,
    build_projector_camera_calib_points_from_charuco,
    detect_charuco_corners_in_camera,
    draw_charuco_detection,
    ensure_aruco,
    validate_charuco_mode,
)

CHARUCO_MODE_FULLSCREEN = _charuco.CHARUCO_MODE_FULLSCREEN
CHARUCO_MODE_ALIGN_BOX = _charuco.CHARUCO_MODE_ALIGN_BOX


def _sync_charuco_config() -> None:
    _charuco.CHARUCO_MODE_FULLSCREEN = bool(CHARUCO_MODE_FULLSCREEN)
    _charuco.CHARUCO_MODE_ALIGN_BOX = bool(CHARUCO_MODE_ALIGN_BOX)
    _charuco.ALIGN_CHARUCO_RECT = ALIGN_CHARUCO_RECT


def generate_charuco_projector_assets(out_dir: str | Path, spec: CharucoSpec = CharucoSpec()):
    _sync_charuco_config()
    return _charuco.generate_charuco_projector_assets(out_dir, spec)


def main() -> None:
    from homography.detect_linux import main as run_detect

    run_detect()


if __name__ == "__main__":
    main()
