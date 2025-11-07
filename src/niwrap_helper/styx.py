"""Styx-related functions."""

import logging
import os
import shutil
from pathlib import Path
from typing import Literal, overload

import niwrap
import yaml

from .types import (
    BaseRunner,
    DockerType,
    LocalType,
    SingularityType,
    StrPath,
)


@overload
def setup_styx() -> tuple[logging.Logger, niwrap.LocalRunner]: ...


@overload
def setup_styx(
    runner: DockerType,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, StrPath] | None,
    graph_runner: Literal[False],
    *args,
    **kwargs,
) -> tuple[logging.Logger, niwrap.DockerRunner]: ...


@overload
def setup_styx(
    runner: SingularityType,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, StrPath] | None,
    graph_runner: Literal[False],
    *args,
    **kwargs,
) -> tuple[logging.Logger, niwrap.SingularityRunner]: ...


@overload
def setup_styx(
    runner: LocalType,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, StrPath] | None,
    graph_runner: Literal[False],
    *args,
    **kwargs,
) -> tuple[logging.Logger, niwrap.LocalRunner]: ...


@overload
def setup_styx(
    runner: str,
    tmp_env: str,
    tmp_dir: str,
    image_map: StrPath | dict[str, StrPath] | None,
    graph_runner: Literal[True],
    *args,
    **kwargs,
) -> tuple[logging.Logger, niwrap.GraphRunner]: ...


def setup_styx(
    runner: str = "local",
    tmp_env: str = "LOCAL",
    tmp_dir: str = "styx_tmp",
    image_map: StrPath | dict[str, StrPath] | None = None,
    graph_runner: bool = False,
    *args,
    **kwargs,
) -> tuple[logging.Logger, BaseRunner | niwrap.GraphRunner]:
    """Setup Styx runner.

    Args:
        runner: Type of StyxRunner to use - choices include
            ['local', 'docker', 'podman', 'singularity', 'apptainer']
        tmp_env: Environment variable to query for temporary folder. Defaults: 'LOCAL'
        tmp_dir: Working directory to output to. Defaults: '{tmp_env}/tmp_dir'
        image_map: Path to config file or dictionary containing container mappings to
            disk.
        graph_runner: Flag to make use of GraphRunner middleware.

    Returns:
        A 2-tuple where the first element is the configured logger instance and the
        second is the initialized runner, optionally wrapped in GraphRunner.

    Raises:
        ValueError: if StyxRunner is not set.
    """
    images = (
        yaml.safe_load(Path(image_map).read_text())
        if isinstance(image_map, (str, Path))
        else image_map
    )
    match runner_exec := runner.lower():
        case "docker" | "podman":
            niwrap.use_docker(
                docker_executable=runner_exec,
                image_overrides=images,
                *args,
                **kwargs,
            )
        case "singularity" | "apptainer":
            if images is None:
                raise ValueError("No container mapping provided")
            niwrap.use_singularity(
                singularity_executable=runner_exec, images=images, *args, **kwargs
            )
        case _:
            niwrap.use_local(*args, **kwargs)

    styx_runner = niwrap.get_global_runner()
    styx_runner.data_dir = Path(os.getenv(tmp_env, "/tmp")) / tmp_dir
    logger = logging.getLogger(styx_runner.logger_name)
    logger.setLevel(logging.INFO)
    if graph_runner:
        niwrap.use_graph(styx_runner)
        styx_runner = niwrap.get_global_runner()
    return logger, styx_runner


def gen_hash() -> str:
    """Generate hash for styx runner.

    Returns:
        str: Unique id + incremented execution counter as a hash string.
    """
    runner = niwrap.get_global_runner()
    base_runner = runner.base if isinstance(runner, niwrap.GraphRunner) else runner
    base_runner.execution_counter += 1
    return f"{base_runner.uid}_{base_runner.execution_counter - 1}"


def cleanup() -> None:
    """Clean up after completing run."""
    runner = niwrap.get_global_runner()
    base_runner = runner.base if isinstance(runner, niwrap.GraphRunner) else runner
    base_runner.execution_counter = 0
    shutil.rmtree(base_runner.data_dir)


def save(files: Path | list[Path], out_dir: Path) -> None:
    """Copy niwrap outputted file(s) to specified output directory.

    Args:
        files: Path or list of paths to save.
        out_dir: Output directory to save file(s) to
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure `files` is iterable and process each one
    for file in [files] if isinstance(files, (str, Path)) else files:
        shutil.copy2(file, out_dir / Path(file).name)
