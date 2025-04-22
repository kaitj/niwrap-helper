"""Utilities for working with SLURM."""

import os

from pandas import DataFrame


def get_subject(bids_table: DataFrame) -> str:
    """Get subject from BIDSTable and slurm array task id."""
    array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    subjects = bids_table.flat["sub"].sort_values().unique().tolist()

    if array_id:
        return subjects[int(array_id)]
    elif len(subjects) > 0:
        print("No array id provided - using first subject found")
        return subjects[0]
    else:
        raise ValueError("No array id found")
