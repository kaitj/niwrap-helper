"""Styx-associated helpers."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal, overload, NamedTuple

import niwrap
from styxpodman import PodmanRunner

_LOG_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG)

_RUNNER_LITERAL = Literal["local", "docker", "singularity", "apptainer", "podman"]

class StyxContext(NamedTuple):
    """Styx execution context with logger, runner, and verbosity."""
    logger: logging.Logger
    runner: niwrap.Runner
    verbose: bool

def setup_styx(
    runner: _RUNNER_LITERAL = "local",
    tmp_dir: str | Path | None = None,
    image_overrides: dict[str, str] | None = None,
    graph: bool = False,
    verbose: int = 0
    **kwargs,
) -> tuple[logging.Logger, BaseRunner | niwrap.GraphRunner]:
    """Set up Styx with the appropriate runner for NiWrap.

    Args:
        runner: Container/execution backend.  One of
            ``'local'``, ``'apptainer'``, ``'docker'``,
            ``'podman'``, or ``'singularity'``.
        tmp_dir: Parent directory for the temporary working directory.
            Defaults to the system temp directory.
        image_overrides: Optional mapping of tool name → container image tag.
        graph: When ``True``, wrap the runner in a
            :class:`niwrap.GraphRunner` middleware.
        verbose: Verbosity level (0 = WARNING, 1 = INFO, 2+ = DEBUG).
        **kwargs: Extra keyword arguments forwarded to the runner constructor.

    Returns:
        :class:`StyxContext` containing the configured logger, runner, and
        a boolean flag indicating whether verbose output is active.

    Raises:
        NotImplementedError: For unrecognized ``runner`` values.
    """
    match runner_exec := runner.lower():
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
                singularity_executable="singularity",
                image_overrides=image_overrides,
                **kwargs,
            )
        case _:
            raise NotImplementedError(
                f"Unknown runner selection '{runner}' - please select one of "
                "'local', 'apptainer', 'docker', 'podman', or 'singularity'"
            )

    styx_runner = niwrap.get_global_runner()
    styx_runner.data_dir = Path(tempfile.mkdtemp(dir=tmp_dir))

    logger = logging.getLogger(styx_runner.logger_name)
    logger.setLevel(_LOG_LEVELS[min(verbose, len(_LOG_LEVELS) - 1)])

    if graph:
        niwrap.use_graph(styx_runner)
        styx_runner = niwrap.get_global_runner()

    return StyxContext(logger=logger, runner=styx_runner, verbose=verbose > 0)

def _get_base_runner() -> niwrap.StyxRunner:
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
    items: list[Path] = [files] if isinstance(files, (str, Path)) else list(files)
    for file in items:
        shutil.copy2(file, out_dir / Path(file).name)