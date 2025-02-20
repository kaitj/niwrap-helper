"""Custom types."""

from pathlib import Path
from typing import Literal, TypeAlias

from styxdefs import LocalRunner
from styxdocker import DockerRunner
from styxsingularity import SingularityRunner

StrPath = str | Path
StyxRunner = LocalRunner | DockerRunner | SingularityRunner


DockerType: TypeAlias = Literal["docker", "Docker", "DOCKER"]
SingularityType: TypeAlias = Literal[
    "singularity", "Singularity", "SINGULARITY", "apptainer", "Apptainer", "APPTAINER"
]
LocalType: TypeAlias = Literal["local", "Local", "LOCAL"]
