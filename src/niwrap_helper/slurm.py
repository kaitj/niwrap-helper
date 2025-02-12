"""Utilities for working with SLURM."""

import os

from pandas import DataFrame


def get_subject(bids_table: DataFrame) -> str:
    """Get subject from BIDSTable and slurm array task id."""
    array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    subjects = bids_table.flat["sub"].sort_values().unique().tolist()

    if array_id is not None:
        return subjects[int(array_id)]
    else:
        raise ValueError("No array id found")
