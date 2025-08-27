"""Custom types."""

from typing import Literal, TypeAlias

from bids2table._pathlib import PathT
from niwrap import DockerRunner, LocalRunner, SingularityRunner

StrPath = str | PathT
BaseRunner = DockerRunner | LocalRunner | SingularityRunner

DockerType: TypeAlias = Literal["docker", "Docker", "DOCKER"]
SingularityType: TypeAlias = Literal[
    "singularity", "Singularity", "SINGULARITY", "apptainer", "Apptainer", "APPTAINER"
]
LocalType: TypeAlias = Literal["local", "Local", "LOCAL"]
