"""Utility functions for working with BIDS-associated objects."""

from pathlib import Path
from typing import Literal, overload

import bids2table as b2t
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from niwrap_helper.types import StrPath


def get_bids_table(
    dataset_dir: StrPath,
    index: StrPath = ".index.b2t",
) -> pd.DataFrame:
    """Get and return BIDSTable for a given dataset."""
    dataset_dir = Path(dataset_dir)

    # Get table
    if (index_fp := (dataset_dir / index)).exists():
        table: pa.Table = pq.read_table(index_fp)
    else:
        tables = b2t.batch_index_dataset(b2t.find_bids_datasets(dataset_dir))
        table: pa.Table = pa.concat_tables(tables)  # type: ignore[no-redef]

    # Normalize extra entities column
    extra_entities_df = pd.json_normalize(table.column("extra_entities").to_pylist())
    table = table.drop(["extra_entities"])

    return pd.concat([table.to_pandas(), extra_entities_df], axis=1)


@overload
def bids_path(
    directory: Literal[False], return_path: Literal[False], **entities
) -> str: ...


@overload
def bids_path(
    directory: Literal[True], return_path: Literal[False], **entities
) -> Path: ...


@overload
def bids_path(
    directory: Literal[False], return_path: Literal[True], **entities
) -> Path: ...


def bids_path(
    directory: bool = False, return_path: bool = False, **entities
) -> StrPath:
    """Generate BIDS name / path."""
    if directory and return_path:
        raise ValueError("Only one of 'directory' or 'return_path' can be True")
    name = b2t.format_bids_path(entities)
    return name.parent if directory else name if return_path else name.name
