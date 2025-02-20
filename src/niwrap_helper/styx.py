"""Styx-related functions."""

import logging
import os
import shutil
from pathlib import Path
from typing import overload

import yaml
from styxdefs import LocalRunner, OutputPathType, set_global_runner
from styxdocker import DockerRunner
from styxsingularity import SingularityRunner

from niwrap_helper.types import (
    DockerType,
    LocalType,
    SingularityType,
    StrPath,
    StyxRunner,
)


@overload
def setup_styx(
    image_map: StrPath | None,
    runner: DockerType,
    tmp_env: str = "LOCAL",
    tmp_dir: str = "styx_tmp",
) -> tuple[logging.Logger, DockerRunner]: ...


@overload
def setup_styx(
    image_map: StrPath | None,
    runner: SingularityType,
    tmp_env: str = "LOCAL",
    tmp_dir: str = "styx_tmp",
) -> tuple[logging.Logger, SingularityRunner]: ...


@overload
def setup_styx(
    image_map: StrPath | None,
    runner: LocalType,
    tmp_env: str = "LOCAL",
    tmp_dir: str = "styx_tmp",
) -> tuple[logging.Logger, LocalRunner]: ...


def setup_styx(
    image_map: StrPath | None = None,
    runner: str = "local",
    tmp_env: str = "LOCAL",
    tmp_dir: str = "styx_tmp",
) -> tuple[logging.Logger, StyxRunner]:
    """Setup Styx runner."""
    runner = runner.lower()
    if runner == "docker":
        styx_runner = DockerRunner()
    elif runner == "singularity" or runner == "apptainer":
        if image_map is None:
            raise ValueError("No container mapping provided")
        styx_runner = SingularityRunner(
            images=yaml.safe_load(Path(image_map).read_text())
        )
    else:
        styx_runner = LocalRunner()
    styx_runner.data_dir = Path(os.getenv(tmp_env, "/tmp")) / tmp_dir

    set_global_runner(styx_runner)

    return logging.getLogger(styx_runner.logger_name), styx_runner


def gen_hash(runner: StyxRunner) -> str:
    """Generate hash for styx runner."""
    runner.execution_counter += 1
    return f"{runner.uid}_{runner.execution_counter - 1}"


def cleanup(runner: StyxRunner) -> None:
    """Clean up after completing run."""
    shutil.rmtree(runner.data_dir)
    runner.execution_counter = 0


def save(
    files: OutputPathType | list[OutputPathType],
    out_dir: Path,
) -> None:
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
