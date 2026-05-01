"""Styx-associated helpers."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal, NamedTuple

import niwrap
from styxpodman import PodmanRunner

_LOG_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG)

RunnerType = Literal["local", "docker", "podman", "apptainer", "singularity"]

_RUNNER_EXECUTABLES: list[tuple[RunnerType, list[str]]] = [
    ("docker", ["docker"]),
    ("podman", ["podman"]),
    ("singularity", ["apptainer", "singularity"]),
]


class StyxContext(NamedTuple):
    """Styx execution context with logger and runner."""

    logger: logging.Logger
    runner: niwrap.Runner
    verbose: bool


def resolve_runner(
    runner: RunnerType | Literal["auto"] = "auto",
) -> tuple[RunnerType, str]:
    """Resolve runner selection, auto-detecting if needed.

    When runner is "auto", checks for available container runtimes on PATH
    in order of preference: docker > podman > apptainer/singularity > local.

    Args:
        runner: Runner type or "auto" for auto-detection.

    Returns:
        Tuple of (runner_type, executable_name).
    """
    if runner != "auto":
        return runner, runner

    for runner_type, executables in _RUNNER_EXECUTABLES:
        for exe in executables:
            if shutil.which(exe):
                return runner_type, exe
    return "local", "local"


def setup_runner(
    runner: RunnerType | Literal["auto"] = "auto",
    tmp_dir: str | Path | None = None,
    image_overrides: dict[str, str] | None = None,
    graph: bool = False,  # noqa: FBT001, FBT002 - graph runner flag
    verbose: int = 0,
    **kwargs,  # noqa: ANN003 - kwargs for runners
) -> StyxContext:
    """Set up Styx with the appropriate runner for NiWrap.

    Args:
        runner: Type of runner to use. "auto" detects the first available
            container runtime, falling back to "local".
        tmp_dir: Working directory to output to
        image_overrides: Dictionary containing overrides for container tags.
        graph: When ``True``, wrap the runner in a
            :class:`niwrap.GraphRunner` middleware.
        verbose: Verbosity level (0 = WARNING, 1 = INFO, 2+ = DEBUG).
        **kwargs: Additional keyword arguments passed for runner setup.

    Returns:
        :class:`StyxContext` containing the configured logger, runner, and
        a boolean flag indicating whether verbose output is active.

    Raises:
        NotImplementedError: For unrecognized ``runner`` values.
    """
    runner_type, runner_exec = resolve_runner(runner)

    match runner_type:
        case "local":
            niwrap.use_local()
        case "docker":
            niwrap.use_docker(
                docker_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case "podman":
            niwrap.set_global_runner(
                runner=PodmanRunner(
                    podman_executable=runner_exec,
                    image_overrides=image_overrides,
                    podman_user_id=0,
                    **kwargs,
                )
            )
        case "apptainer" | "singularity":
            niwrap.use_singularity(
                singularity_executable=runner_exec,
                image_overrides=image_overrides,
                **kwargs,
            )
        case _:
            raise NotImplementedError(
                f"Unknown runner selection '{runner}' - please select one of "
                "'auto', 'local', 'docker', 'podman', or 'singularity'"
            )

    styx_runner = niwrap.get_global_runner()
    if tmp_dir is not None:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    styx_runner.data_dir = Path(tempfile.mkdtemp(dir=tmp_dir))

    styx_logger = logging.getLogger(styx_runner.logger_name)
    styx_logger.setLevel(_LOG_LEVELS[min(verbose, len(_LOG_LEVELS) - 1)])

    if graph:
        niwrap.use_graph(styx_runner)
        styx_runner = niwrap.get_global_runner()

    return StyxContext(logger=styx_logger, runner=styx_runner, verbose=verbose > 0)


def _get_base_runner() -> niwrap.Runner | niwrap.GraphRunner:
    """Unwrap GraphRunner middleware to retrieve the underlying base runner."""
    runner = niwrap.get_global_runner()
    return runner.base if isinstance(runner, niwrap.GraphRunner) else runner


def generate_exec_folder(suffix: str = "python") -> Path:
    """Generate an execution folder following the Styx hash pattern.

    Args:
        suffix: Label appended to the folder name (default: ``'python'``).

    Returns:
        :class:`~pathlib.Path` to the newly created execution folder.
    """
    base = _get_base_runner()
    dir_path = Path(base.data_dir) / f"{base.uid}_{base.execution_counter}_{suffix}"
    dir_path.mkdir(parents=True)
    base.execution_counter += 1
    return dir_path


def cleanup_session() -> None:
    """Clean up temporary data after completing a NiWrap session."""
    base = _get_base_runner()
    base.execution_counter = 0
    shutil.rmtree(base.data_dir, ignore_errors=True)


def save(files: Path | list[Path], out_dir: Path) -> None:
    """Copy NiWrap-output file(s) to the specified directory.

    Args:
        files: A single path or a list of paths to copy.
        out_dir: Destination directory (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[Path] = [files] if isinstance(files, (str | Path)) else list(files)
    for file in items:
        shutil.copy2(file, out_dir / Path(file).name)
