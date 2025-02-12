"""Custom types."""

from pathlib import Path

from styxdefs import LocalRunner
from styxdocker import DockerRunner
from styxsingularity import SingularityRunner

StrPath = str | Path
StyxRunner = LocalRunner | DockerRunner | SingularityRunner
