"""Utility functions for working with BIDS-associated objects."""

from typing import Literal, overload

import bids2table as b2t
import pyarrow as pa
import pyarrow.parquet as pq
from bids2table._pathlib import as_path

from .types import PathT, StrPath, StrPathT


def get_bids_table(
    dataset_dir: StrPathT,
    b2t_index: StrPath | None = None,
    max_workers: int | None = 0,
    verbose: bool = False,
) -> pa.Table:
    """Get and return BIDSTable for a given dataset.

    Args:
        dataset_dir: Path to dataset directory.
        b2t_index: Path to bids2table parquet table. If provided and the file exists
            the parquet table will be used. If 'None' or the file does not exist, the
            dataset directory will be indexed.
        max_workers: (bids2table) Number of indexing processes to run in parallel.
            Setting `max_workers=0` (the default) uses the main process only. Setting
            `max_workers=None` starts as many workers as there are available CPUs. See
            `concurrent.futures.ProcessPoolExecutor` for details.
        verbose: Show verbose messages.

    Returns:
        pyarrow.Table: A concatenated Arrow table index for all BIDS datasets.
    """
    ds_path = as_path(dataset_dir)

    # Load / generate table
    b2t_fp = ds_path / b2t_index if b2t_index else None
    if b2t_fp and b2t_fp.exists():
        table = pq.read_table(b2t_fp)
    else:
        tables = b2t.batch_index_dataset(
            b2t.find_bids_datasets(ds_path),  # type: ignore
            max_workers=max_workers,
            show_progress=verbose,
        )
        table = pa.concat_tables(tables)

    if "extra_entities" not in table.column_names:
        return table

    # Expand "extra_entities"
    extra_entities = table["extra_entities"].to_pylist()
    extra_entities_dicts = []
    all_keys: set = set()
    for ent in extra_entities:
        d = dict(ent) if isinstance(ent, list) else {}
        all_keys.update(d.keys())
        extra_entities_dicts.append(d)
    if not all_keys:
        return table
    all_keys = set(sorted(all_keys))
    rows = [{k: d.get(k, None) for k in all_keys} for d in extra_entities_dicts]
    extra_entities_table = pa.Table.from_pylist(rows).append_column(
        "path", table["path"]
    )
    # Conditional sorting to ensure deterministic join
    paths_main = table["path"].to_pylist()
    paths_extra = extra_entities_table["path"].to_pylist()
    if paths_main != paths_extra:
        table = table.sort_by([("path", "ascending")])
        extra_entities_table = extra_entities_table.sort_by([("path", "ascending")])
    # Append only new columns
    existing = set(table.column_names)
    table = table.drop(["extra_entities"])
    for name in extra_entities_table.column_names:
        if name != "path" and name not in existing:
            table = table.append_column(name, extra_entities_table[name])
    return table


@overload
def bids_path(**entities) -> str: ...


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
) -> StrPathT:
    """Generate BIDS name / path.

    Args:
        directory: Flag to return only parent directories - mutually exclusive with
            'return_path'. If both set to false, only returns the file name.
        return_path: Flag to return full path - mutually exclusive with 'directory'.
            If both set to false, only returns the file name.
        **entities: BIDS-entities provided as keyword arguments to be used for
            formulating the BIDS filename / filepath.

    Returns:
        str: A BIDS-formatted filename or filepath.
    """
    if directory and return_path:
        raise ValueError("Only one of 'directory' or 'return_path' can be True")
    name = b2t.format_bids_path(entities)
    return name.parent if directory else name if return_path else name.name
