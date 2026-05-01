# Pipeline entrypoint for the GUI-driven workflow.

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from homography.detect_linux import main as run_detect

ProgressCallback = Callable[[int, int, str], None]

LOG_PATH = Path(__file__).resolve().parent / "startup.log"


@dataclass
class PipelineStep:
    name: str
    runner: Callable[["ProgressReporter"], None]
    step_count: int = 1


class ProgressReporter:
    """Sends pipeline progress updates back to the GUI."""

    def __init__(self, total_steps: int, callback: ProgressCallback | None = None):
        self.total_steps = max(0, total_steps)
        self.current_step = 0
        self._callback = callback

    def emit(self, description: str, step: int | None = None) -> None:
        if step is not None:
            self.current_step = max(0, min(step, self.total_steps))
        if self._callback is not None:
            self._callback(self.current_step, self.total_steps, description)

    def start_step(self, step_number: int, description: str) -> None:
        self.emit(description, step=step_number)

    def detail(self, description: str) -> None:
        self.emit(description)

    def section(self, start_step: int, step_count: int) -> "StepProgressReporter":
        return StepProgressReporter(
            parent=self,
            start_step=start_step,
            step_count=max(1, step_count),
        )


class StepProgressReporter:
    """Scoped progress reporter for one runner's portion of the global progress bar."""

    def __init__(self, parent: ProgressReporter, start_step: int, step_count: int):
        self.parent = parent
        self.start_step = start_step
        self.step_count = max(1, step_count)

    def step(self, local_step: int, description: str) -> None:
        bounded_step = max(1, min(local_step, self.step_count))
        global_step = self.start_step + bounded_step - 1
        self.parent.emit(description, step=global_step)

    def detail(self, description: str) -> None:
        self.parent.detail(description)


def _configure_logging() -> None:
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _build_pipeline() -> list[PipelineStep]:
    return [
        PipelineStep(
            name="Detect and project pipeline.", 
            runner=run_detect, 
            step_count=7),
    ]


def main(progress_callback: ProgressCallback | None = None) -> None:
    _configure_logging()

    steps = _build_pipeline()
    total_steps = sum(step.step_count for step in steps)
    reporter = ProgressReporter(total_steps=total_steps, callback=progress_callback)
    reporter.emit("Preparing pipeline...", step=0)
    logging.info("Pipeline started with %s displayed step(s).", total_steps)

    next_step_index = 1
    for step in steps:
        runner_progress = reporter.section(
            start_step=next_step_index,
            step_count=step.step_count,
        )
        runner_progress.step(1, step.name)
        logging.info(
            "Starting pipeline section at step %s (%s slots): %s",
            next_step_index,
            step.step_count,
            step.name,
        )
        step.runner(runner_progress)
        next_step_index += step.step_count

    reporter.emit("Complete!", step=total_steps)
    logging.info("Pipeline completed successfully.")