"""Utility functions for working with BIDS-associated objects."""

from typing import Literal, overload

import bids2table as b2t
import pyarrow as pa
import pyarrow.parquet as pq
from bids2table._pathlib import PathT, as_path
from pandas import DataFrame

from niwrap_helper.types import StrPath


@overload
def get_bids_table(
    dataset_dir: StrPath, index: StrPath, return_type: Literal["pandas"]
) -> DataFrame: ...


@overload
def get_bids_table(
    dataset_dir: StrPath, index: StrPath, return_type: Literal["pyarrow"]
) -> pa.Table: ...


def get_bids_table(
    dataset_dir: StrPath,
    index: StrPath | None = ".index.b2t",
    return_type: Literal["pandas"] | Literal["pyarrow"] = "pandas",
) -> pa.Table | DataFrame:
    """Get and return BIDSTable for a given dataset."""
    dataset_dir = as_path(dataset_dir)

    # Get table
    if index is not None and (index_fp := (dataset_dir / index)).exists():
        table = pq.read_table(index_fp)
    else:
        tables = b2t.batch_index_dataset(b2t.find_bids_datasets(dataset_dir))  # type: ignore
        table = pa.concat_tables(tables)

    # Set each extra entity as separate column
    extra_entities = table.column("extra_entities").to_pylist()
    extra_entities_dicts = [
        dict(pairs) if isinstance(pairs, list) else {} for pairs in extra_entities
    ]
    all_keys = set().union(*(d.keys() for d in extra_entities_dicts if d))

    if all_keys:
        extra_entities_dicts = [
            {k: d.get(k) for k in all_keys} for d in extra_entities_dicts
        ]
        extra_entities_table = pa.Table.from_pylist(extra_entities_dicts)
        table = pa.concat_tables(
            [table, extra_entities_table], promote_options="default"
        )
    table = table.drop(["extra_entities"])

    dispatch = {"pandas": table.to_pandas, "pyarrow": lambda: table}
    try:
        return dispatch[return_type]()
    except KeyError:
        raise ValueError(f"Unsupported return_type: {return_type}")


@overload
def bids_path(
    directory: Literal[False], return_path: Literal[False], **entities
) -> str: ...


@overload
def bids_path(
    directory: Literal[True], return_path: Literal[False], **entities
) -> PathT: ...


@overload
def bids_path(
    directory: Literal[False], return_path: Literal[True], **entities
) -> PathT: ...


def bids_path(
    directory: bool = False, return_path: bool = False, **entities
) -> StrPath:
    """Generate BIDS name / path."""
    if directory and return_path:
        raise ValueError("Only one of 'directory' or 'return_path' can be True")
    name = b2t.format_bids_path(entities)
    return name.parent if directory else name if return_path else name.name
