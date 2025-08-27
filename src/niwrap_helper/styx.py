"""Styx-related functions."""

import logging
import os
import shutil
from pathlib import Path
from typing import Literal, overload

import yaml
from niwrap import (
    DockerRunner,
    GraphRunner,
    LocalRunner,
    SingularityRunner,
    set_global_runner,
)

from niwrap_helper.types import (
    BaseRunner,
    DockerType,
    LocalType,
    SingularityType,
    StrPath,
)


@overload
def setup_styx() -> tuple[logging.Logger, LocalRunner]: ...


@overload
def setup_styx(
    runner: DockerType,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, str] | None,
    graph_runner: Literal[False],
) -> tuple[logging.Logger, DockerRunner]: ...


@overload
def setup_styx(
    runner: SingularityType,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, str] | None,
    graph_runner: Literal[False],
) -> tuple[logging.Logger, SingularityRunner]: ...


@overload
def setup_styx(
    runner: LocalType,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, str] | None,
    graph_runner: Literal[False],
) -> tuple[logging.Logger, LocalRunner]: ...


@overload
def setup_styx(
    runner: str,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, str] | None,
    graph_runner: Literal[True],
) -> tuple[logging.Logger, GraphRunner]: ...


def setup_styx(
    runner: str = "local",
    tmp_env: str = "LOCAL",
    tmp_dir: str = "styx_tmp",
    image_map: StrPath | dict[str, str] | None = None,
    graph_runner: bool = False,
) -> tuple[logging.Logger, BaseRunner | GraphRunner]:
    """Setup Styx runner."""
    match runner.lower():
        case "docker":
            styx_runner = DockerRunner()
        case "singularity" | "apptainer":
            if isinstance(image_map, (str, Path)):
                styx_runner = SingularityRunner(
                    images=yaml.safe_load(Path(image_map).read_text())
                )
            elif isinstance(image_map, dict):
                styx_runner = SingularityRunner(images=image_map)
            else:
                raise ValueError("No container mapping provided")
        case _:
            styx_runner = LocalRunner()

    logger_name = styx_runner.logger_name
    styx_runner.data_dir = Path(os.getenv(tmp_env, "/tmp")) / tmp_dir
    if graph_runner:
        styx_runner = GraphRunner(styx_runner)
    set_global_runner(styx_runner)

    return logging.getLogger(logger_name), styx_runner


def _get_base_runner(runner: BaseRunner | GraphRunner) -> BaseRunner:
    """Return base styx runner used."""
    if isinstance(runner, GraphRunner):
        return runner.base
    return runner


def gen_hash(runner: BaseRunner | GraphRunner) -> str:
    """Generate hash for styx runner."""
    base_runner = _get_base_runner(runner=runner)
    base_runner.execution_counter += 1
    return f"{base_runner.uid}_{base_runner.execution_counter - 1}"


def cleanup(runner: BaseRunner | GraphRunner) -> None:
    """Clean up after completing run."""
    base_runner = _get_base_runner(runner=runner)
    base_runner.execution_counter = 0
    shutil.rmtree(base_runner.data_dir)


def save(files: Path | list[Path], out_dir: Path) -> None:
    """Save file(s) to specified output directory, preserving directory structure."""

    def _save_file(fpath: Path) -> None:
        """Save individual file, preserving directory structure."""
        for part in fpath.parts:
            if part.startswith("sub-"):
                out_fpath = out_dir.joinpath(*fpath.parts[fpath.parts.index(part) :])
                out_fpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, out_fpath)
                return
        raise ValueError(f"Unable to find relevant file path components for {fpath}")

    # Ensure `files` is iterable and process each one
    for file in [files] if isinstance(files, (str, Path)) else files:
        _save_file(Path(file))
