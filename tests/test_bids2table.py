"""Tests for niwrap_helper.bids2table."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import polars as pl
import polars.type_aliases as pl_types
import pytest

if TYPE_CHECKING:
    import pyarrow as pa

# ---------------------------------------------------------------------------
# Shared helpers / factories
# ---------------------------------------------------------------------------

_EXTRA_SCHEMA = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))

_BASE_ROW: dict[str, Any] = {
    "sub": "01",
    "ses": "baseline",
    "datatype": "anat",
    "suffix": "T1w",
    "desc": None,
    "ext": ".nii.gz",
    "task": None,
    "run": None,
    "space": None,
    "root": "/data",
    "path": "sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
    "extra_entities": [],
}

_DF_SCHEMA: dict[str, pl_types.DataType] = {
    "sub": pl.String,
    "ses": pl.String,
    "datatype": pl.String,
    "suffix": pl.String,
    "desc": pl.String,
    "ext": pl.String,
    "task": pl.String,
    "run": pl.Int64,
    "space": pl.String,
    "root": pl.String,
    "path": pl.String,
    "extra_entities": _EXTRA_SCHEMA,
}


def _row(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401 - internal helper
    return {**_BASE_ROW, **overrides}


def _df(*rows: dict[str, Any]) -> pl.DataFrame:
    """Build a typed Polars DataFrame from one or more row dicts."""
    if not rows:
        rows = (_BASE_ROW,)
    cols: dict[str, list[Any]] = {k: [r[k] for r in rows] for k in rows[0]}
    return pl.DataFrame(cols, schema=_DF_SCHEMA)


def _arrow(*rows: dict[str, Any]) -> pa.Table:
    """Build a PyArrow table via _df (keeps schema consistent)."""
    return _df(*rows).to_arrow()


# ---------------------------------------------------------------------------
# load_table
# ---------------------------------------------------------------------------


class TestLoadTable:
    """Tests for load_table."""

    def test_reads_parquet_and_skips_indexing(self, tmp_path: Path) -> None:
        """index_fpath reads parquet directly and never calls batch_index_dataset."""
        import bids2table as b2t

        from niwrap_helper.bids2table import load_table

        parquet = tmp_path / "index.parquet"
        expected = _df(_row())
        expected.write_parquet(parquet)

        with patch.object(b2t, "batch_index_dataset") as mock_index:
            result = load_table("/ignored", index_fpath=parquet)

        mock_index.assert_not_called()
        assert isinstance(result, pl.DataFrame)
        assert result.shape == expected.shape

    @pytest.mark.parametrize(
        ("n_tables", "expected_rows"),
        [(1, 1), (2, 2)],
        ids=["single", "multi-concat"],
    )
    def test_indexes_and_concatenates(self, n_tables: int, expected_rows: int) -> None:
        """batch_index_dataset results are concatenated into one DataFrame."""
        import bids2table as b2t

        from niwrap_helper.bids2table import load_table

        arrows = [_arrow(_row(sub=str(i).zfill(2))) for i in range(n_tables)]

        with (
            patch.object(b2t, "find_bids_datasets", return_value=["ds"] * n_tables),
            patch.object(b2t, "batch_index_dataset", return_value=arrows),
        ):
            result = load_table("/some/dir")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == expected_rows

    def test_raises_value_error_when_no_datasets(self) -> None:
        """ValueError raised when no tables are returned."""
        import bids2table as b2t

        from niwrap_helper.bids2table import load_table

        with (
            patch.object(b2t, "find_bids_datasets", return_value=[]),
            patch.object(b2t, "batch_index_dataset", return_value=[]),
            pytest.raises(ValueError, match="No datasets found"),
        ):
            load_table("/empty/dir")

    def test_raises_type_error_on_non_dataframe(self) -> None:
        """TypeError raised when from_arrow returns something other than DataFrame."""
        import bids2table as b2t

        from niwrap_helper.bids2table import load_table

        with (
            patch.object(b2t, "find_bids_datasets", return_value=["ds1"]),
            patch.object(b2t, "batch_index_dataset", return_value=[_arrow(_row())]),
            patch("polars.from_arrow", return_value=MagicMock(spec=[])),
            pytest.raises(TypeError, match="Expected DataFrame"),
        ):
            load_table("/some/dir")

    @pytest.mark.parametrize(
        ("kwarg", "value", "batch_kwarg"),
        [
            ("verbose", True, "show_progress"),
            ("max_workers", 4, "max_workers"),
        ],
    )
    def test_kwargs_forwarded_to_batch_index(
        self,
        kwarg: str,
        value: Any,  # noqa: ANN401 - multi uses
        batch_kwarg: str,
    ) -> None:
        """'verbose' and 'max_workers' are forwarded to batch_index_dataset."""
        import bids2table as b2t

        from niwrap_helper.bids2table import load_table

        with (
            patch.object(b2t, "find_bids_datasets", return_value=["ds1"]),
            patch.object(
                b2t, "batch_index_dataset", return_value=[_arrow(_row())]
            ) as mock_idx,
        ):
            load_table("/some/dir", **{kwarg: value})

        assert mock_idx.call_args[1][batch_kwarg] == value


# ---------------------------------------------------------------------------
# get_extra_entity
# ---------------------------------------------------------------------------


class TestGetExtraEntity:
    """Tests for get_extra_entity."""

    @pytest.fixture
    def extra_df(self) -> pl.DataFrame:
        """DataFrame with extra entities."""
        return pl.DataFrame(
            {"extra_entities": [[{"key": "from", "value": "T1w"}]]},
            schema={"extra_entities": _EXTRA_SCHEMA},
        )

    def test_extracts_matching_key(self, extra_df: pl.DataFrame) -> None:
        """Test keys match returns row."""
        from niwrap_helper.bids2table import get_extra_entity

        assert extra_df.select(get_extra_entity("from")).to_series().to_list() == [
            "T1w"
        ]

    def test_missing_key_returns_none(self, extra_df: pl.DataFrame) -> None:
        """Test missing key returns none."""
        from niwrap_helper.bids2table import get_extra_entity

        assert extra_df.select(get_extra_entity("nope")).to_series().to_list() == [None]

    def test_empty_list_returns_none(self) -> None:
        """Test no entities returns None."""
        from niwrap_helper.bids2table import get_extra_entity

        df = pl.DataFrame(
            {"extra_entities": [[]]}, schema={"extra_entities": _EXTRA_SCHEMA}
        )
        assert df.select(get_extra_entity("from")).to_series().to_list() == [None]

    def test_duplicate_key_returns_first(self) -> None:
        """Test duplicate keys returns first."""
        from niwrap_helper.bids2table import get_extra_entity

        df = pl.DataFrame(
            {
                "extra_entities": [
                    [
                        {"key": "from", "value": "first"},
                        {"key": "from", "value": "second"},
                    ]
                ]
            },
            schema={"extra_entities": _EXTRA_SCHEMA},
        )
        assert df.select(get_extra_entity("from")).to_series().to_list() == ["first"]

    def test_usable_in_filter(self) -> None:
        """Test filtered dataframes are usable."""
        from niwrap_helper.bids2table import get_extra_entity

        df = pl.DataFrame(
            {
                "sub": ["01", "02"],
                "extra_entities": [
                    [{"key": "from", "value": "T1w"}],
                    [{"key": "from", "value": "MNI"}],
                ],
            },
            schema={"sub": pl.String, "extra_entities": _EXTRA_SCHEMA},
        )
        filtered = df.filter(get_extra_entity("from") == "T1w")
        assert len(filtered) == 1
        assert filtered["sub"][0] == "01"


# ---------------------------------------------------------------------------
# get_file_path
# ---------------------------------------------------------------------------


class TestGetFilePath:
    """Tests for get_file_path."""

    def test_returns_correct_path_object(self) -> None:
        """Reconstructs root/path and returns a Path."""
        from niwrap_helper.bids2table import get_file_path

        df = _df(_row(root="/data", path="sub-01/ses-baseline/anat/T1w.nii.gz"))
        result = get_file_path(df, sub="01", ses="baseline")

        assert result == Path("/data/sub-01/ses-baseline/anat/T1w.nii.gz")
        assert isinstance(result, Path)

    @pytest.mark.parametrize(
        ("kwargs", "match_override", "other_override"),
        [
            ({"datatype": "func"}, {"datatype": "func"}, {"datatype": "other"}),
            ({"suffix": "bold"}, {"suffix": "bold"}, {"suffix": "other"}),
            ({"desc": "preproc"}, {"desc": "preproc"}, {"desc": "other"}),
            ({"task": "rest"}, {"task": "rest"}, {"task": "other"}),
            (
                {"space": "MNI152NLin2009cAsym"},
                {"space": "MNI152NLin2009cAsym"},
                {"space": "other"},
            ),
            (
                {"run": 1},
                {"run": 1, "path": "run-1.nii.gz"},
                {"run": 2, "path": "run-2.nii.gz"},
            ),
            (
                {"extension": ".nii"},
                {"ext": ".nii.gz", "path": "T1w.nii.gz"},
                {"ext": ".json", "path": "T1w.json"},
            ),
        ],
        ids=["datatype", "suffix", "desc", "task", "space", "run", "extension"],
    )
    def test_filters_by_entity(
        self,
        kwargs: dict[str, Any],
        match_override: dict[str, Any],
        other_override: dict[str, Any],
    ) -> None:
        """Each optional entity narrows results to the correct row."""
        from niwrap_helper.bids2table import get_file_path

        df = _df(_row(**match_override), _row(**other_override))
        result = get_file_path(df, sub="01", ses="baseline", **kwargs)
        assert isinstance(result, Path)

    def test_ses_none_skips_session_filter(self) -> None:
        """Test entities skipped if None."""
        from niwrap_helper.bids2table import get_file_path

        df = _df(_row(ses=None, path="sub-01/anat/T1w.nii.gz"))
        assert isinstance(get_file_path(df, sub="01", ses=None), Path)

    def test_filters_by_extra_entity(self) -> None:
        """Test dataframes can be filtered by extras."""
        from niwrap_helper.bids2table import get_file_path

        df = _df(
            _row(
                path="xfm_from-T1w.h5", extra_entities=[{"key": "from", "value": "T1w"}]
            ),
            _row(
                path="xfm_from-MNI.h5", extra_entities=[{"key": "from", "value": "MNI"}]
            ),
        )
        result = get_file_path(df, sub="01", ses="baseline", extra={"from": "T1w"})
        assert "from-T1w" in str(result)

    @pytest.mark.parametrize(
        ("rows", "sub", "exc", "match"),
        [
            ([], "99", FileNotFoundError, "sub='99'"),
            ([_BASE_ROW, _BASE_ROW, _BASE_ROW], "01", ValueError, "3"),
        ],
        ids=["no-match", "multiple-matches"],
    )
    def test_raises_on_bad_match_count(
        self, rows: list[dict[str, Any]], sub: str, exc: type[Exception], match: str
    ) -> None:
        """FileNotFoundError on zero rows; ValueError on >1 rows."""
        from niwrap_helper.bids2table import get_file_path

        df = (
            _df(*rows)
            if rows
            else pl.DataFrame({k: [] for k in _BASE_ROW}, schema=_DF_SCHEMA)
        )
        with pytest.raises(exc, match=match):
            get_file_path(df, sub=sub, ses="baseline")
