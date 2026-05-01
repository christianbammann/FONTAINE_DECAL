from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np


ENV_DISPLAY_X = "GROUPFILES_PROJECTOR_DISPLAY_X"
ENV_DISPLAY_Y = "GROUPFILES_PROJECTOR_DISPLAY_Y"
ENV_WIDTH = "GROUPFILES_PROJECTOR_WIDTH"
ENV_HEIGHT = "GROUPFILES_PROJECTOR_HEIGHT"
ENV_FULLSCREEN = "GROUPFILES_PROJECTOR_FULLSCREEN"
ENV_SETTLE_MS = "GROUPFILES_PROJECTOR_SETTLE_MS"

DEFAULT_DISPLAY_X = 1920
DEFAULT_DISPLAY_Y = 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ProjectorDisplayConfig:
    width: int
    height: int
    display_x: int = DEFAULT_DISPLAY_X
    display_y: int = DEFAULT_DISPLAY_Y
    fullscreen: bool = True
    settle_ms: int = 250


def load_projector_config(
    width: int,
    height: int,
    *,
    display_x: int = DEFAULT_DISPLAY_X,
    display_y: int = DEFAULT_DISPLAY_Y,
    fullscreen: bool = True,
    settle_ms: int = 250,
) -> ProjectorDisplayConfig:
    return ProjectorDisplayConfig(
        width=max(1, _env_int(ENV_WIDTH, width)),
        height=max(1, _env_int(ENV_HEIGHT, height)),
        display_x=_env_int(ENV_DISPLAY_X, display_x),
        display_y=_env_int(ENV_DISPLAY_Y, display_y),
        fullscreen=_env_bool(ENV_FULLSCREEN, fullscreen),
        settle_ms=max(1, _env_int(ENV_SETTLE_MS, settle_ms)),
    )


def ensure_display_image(
    image: np.ndarray,
    config: ProjectorDisplayConfig,
    *,
    label: str = "projector image",
    resize: bool = False,
) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise RuntimeError(
            f"{label} must be 2D or 3D. Got shape {array.shape}."
        )

    if array.shape[:2] == (int(config.height), int(config.width)):
        return array

    if not resize:
        raise RuntimeError(
            f"{label} must match the configured projector size "
            f"{config.width}x{config.height}, got {array.shape[1]}x{array.shape[0]}."
        )

    interpolation = cv2.INTER_NEAREST if array.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(array, (int(config.width), int(config.height)), interpolation=interpolation)


class ProjectorWindow:
    def __init__(self, window_name: str, config: ProjectorDisplayConfig) -> None:
        self.window_name = str(window_name)
        self.config = config
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return
        logging.info("Opening projector window: namedWindow(%s).", self.window_name)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        logging.info("Opening projector window: moveWindow(%s).", self.window_name)
        cv2.moveWindow(
            self.window_name,
            int(self.config.display_x),
            int(self.config.display_y),
        )
        logging.info("Opening projector window: resizeWindow(%s).", self.window_name)
        cv2.resizeWindow(
            self.window_name,
            int(self.config.width),
            int(self.config.height),
        )
        logging.info("Opening projector window: waitKey(%s).", self.window_name)
        cv2.waitKey(1)
        if self.config.fullscreen:
            logging.info("Opening projector window: setWindowProperty fullscreen(%s).", self.window_name)
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        self._is_open = True
        logging.info("Projector window opened: %s.", self.window_name)

    def show(
        self,
        image: np.ndarray,
        *,
        wait_ms: int = 1,
        resize: bool = False,
        label: str = "projector image",
    ) -> np.ndarray:
        self.open()
        display_image = ensure_display_image(
            image,
            self.config,
            label=label,
            resize=resize,
        )
        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(max(1, int(wait_ms)))
        return display_image

    def wait_for_settle(self, wait_ms: int | None = None) -> None:
        wait_value = self.config.settle_ms if wait_ms is None else int(wait_ms)
        cv2.waitKey(max(1, wait_value))

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            cv2.destroyWindow(self.window_name)
        finally:
            self._is_open = False
