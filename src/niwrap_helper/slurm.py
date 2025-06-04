"""Utilities for working with SLURM."""

import os

import pyarrow as pa
import pyarrow.compute as pc


def get_subject(table: pa.Table) -> str:
    """Get subject from the table and slurm array task id."""
    if isinstance(table, pa.Table):
        raise TypeError("Provided table should be of type 'pyarrow.Table'")

    # Extract subjects and sort
    if "sub" not in table.column_names:
        raise ValueError("'sub' column not found in table")

    sub_column = table.column("sub")
    unique_sub = pc.unique(sub_column)
    sorted_sub = pc.sort_indices(unique_sub)
    sorted_sub = pc.take(unique_sub, sorted_sub)

    subjects = sorted_sub.to_pylist()
    array_id = os.getenv("SLURM_ARRAY_TASK_ID")

    if array_id:
        if int(array_id) > len(subjects):
            raise IndexError(f"{array_id} is out of range")
        return subjects[int(array_id)]
    elif subjects:
        print("No array id provided - using first subject found")
        return subjects[0]
    else:
        raise ValueError("No subject found in 'sub' column")
