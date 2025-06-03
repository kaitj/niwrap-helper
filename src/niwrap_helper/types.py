"""Custom types."""

from typing import Literal, TypeAlias

from bids2table._pathlib import PathT
from styxdefs import LocalRunner
from styxdocker import DockerRunner
from styxsingularity import SingularityRunner

StrPath = str | PathT
StyxRunner = LocalRunner | DockerRunner | SingularityRunner


DockerType: TypeAlias = Literal["docker", "Docker", "DOCKER"]
SingularityType: TypeAlias = Literal[
    "singularity", "Singularity", "SINGULARITY", "apptainer", "Apptainer", "APPTAINER"
]
LocalType: TypeAlias = Literal["local", "Local", "LOCAL"]
