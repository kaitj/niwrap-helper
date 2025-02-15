"""Utility functions for working with BIDS-associated objects."""

from pathlib import Path
from typing import overload

import pandas as pd
from bids2table import BIDSEntities, bids2table

from niwrap_helper.types import StrPath


def get_bids_table(
    dataset_dir: StrPath, dataset_index: str = ".index.b2t", workers: int = 1
) -> pd.DataFrame:
    """Get and return BIDSTable for a given dataset."""
    dataset_dir = Path(dataset_dir)
    b2t = bids2table(
        root=dataset_dir,
        index_path=dataset_dir / dataset_index,
        with_meta=False,
        workers=workers,
    )
    extra_entities = pd.json_normalize(b2t["ent__extra_entities"]).set_index(b2t.index)  # type: ignore
    b2t = pd.concat([b2t, extra_entities.add_prefix("ent__")], axis=1).drop(
        columns="ent__extra_entities"
    )
    return b2t


@overload
def bids_path(
    directory: bool = True, return_path: bool = False, **entities
) -> Path: ...


@overload
def bids_path(
    directory: bool = False, return_path: bool = True, **entities
) -> Path: ...


@overload
def bids_path(
    directory: bool = False, return_path: bool = False, **entities
) -> str: ...


def bids_path(
    directory: bool = False, return_path: bool = False, **entities
) -> StrPath:
    """Generate BIDS name / path."""
    if directory and return_path:
        raise ValueError("Only one of 'directory' or 'return_path' can be True")
    name = BIDSEntities.from_dict(entities).to_path()
    return name.parent if directory else name if return_path else name.name
