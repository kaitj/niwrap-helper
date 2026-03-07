"""Tests for niwrap_helpers.niwrap.py."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------


class _FakeStyxRunner:
    """Sentinel class distinct from niwrap.GraphRunner for isinstance checks."""


def _make_base_runner(
    data_dir: Path | None = None,
    uid: str = "abc123",
    execution_counter: int = 0,
    logger_name: str = "styx",
) -> MagicMock:
    """Return a minimal mock that looks like a StyxRunner.

    Uses spec=_FakeStyxRunner so that isinstance(m, niwrap.GraphRunner)
    returns False without relying on fragile __class__ assignment.
    """
    m = MagicMock(spec=_FakeStyxRunner)
    m.data_dir = data_dir or Path("/tmp/styx_test")  # noqa: S108
    m.uid = uid
    m.execution_counter = execution_counter
    m.logger_name = logger_name
    return m


def _make_graph_runner(base: MagicMock) -> MagicMock:
    """Return a mock GraphRunner wrapping *base*."""
    import niwrap

    m = MagicMock(spec=niwrap.GraphRunner)
    m.base = base
    return m


# ---------------------------------------------------------------------------
# _get_base_runner
# ---------------------------------------------------------------------------


class TestGetBaseRunner:
    """Tests for _get_base_runner."""

    def test_plain_runner_returned_directly(self) -> None:
        """Non-GraphRunner is returned as-is."""
        import niwrap

        from niwrap_helper.niwrap import _get_base_runner

        base = _make_base_runner()
        with patch.object(niwrap, "get_global_runner", return_value=base):
            result = _get_base_runner()

        assert result is base

    def test_graph_runner_unwrapped_to_base(self) -> None:
        """GraphRunner is unwrapped to expose the inner base runner."""
        import niwrap

        from niwrap_helper.niwrap import _get_base_runner

        base = _make_base_runner()
        graph = _make_graph_runner(base)

        with patch.object(niwrap, "get_global_runner", return_value=graph):
            result = _get_base_runner()

        assert result is base


# ---------------------------------------------------------------------------
# setup_styx
# ---------------------------------------------------------------------------


class TestSetupStyx:
    """Tests for setup_styx: runner dispatch, logging levels, graph wrapping."""

    def _run(
        self,
        runner_name: str,
        verbose: int = 0,
        graph: bool = False,  # noqa: FBT001, FBT002 - flag to set up graph runner
        **kwargs,  # noqa: ANN003 - forwarded verbatim to runners
    ) -> tuple[Any, dict[str, MagicMock]]:
        import niwrap

        from niwrap_helper.niwrap import setup_styx

        base = _make_base_runner()

        with (
            patch.object(niwrap, "get_global_runner", return_value=base),
            patch.object(niwrap, "use_local") as mock_local,
            patch.object(niwrap, "use_docker") as mock_docker,
            patch.object(niwrap, "use_singularity") as mock_singularity,
            patch.object(niwrap, "set_global_runner") as mock_set,
            patch.object(niwrap, "use_graph") as mock_graph,
            patch("niwrap_helper.niwrap.PodmanRunner") as mock_podman,
            patch("tempfile.mkdtemp", return_value="/tmp/styx_fake"),  # noqa: S108
        ):
            ctx = setup_styx(
                runner=runner_name,  # type: ignore[arg-type]
                verbose=verbose,
                graph=graph,
                **kwargs,
            )

        return ctx, {
            "local": mock_local,
            "docker": mock_docker,
            "singularity": mock_singularity,
            "set_global_runner": mock_set,
            "use_graph": mock_graph,
            "PodmanRunner": mock_podman,
            "base": base,
        }

    # --- runner dispatch ---

    # Each entry: (runner_name, mock_key, expected_call_kwargs)
    # mock_key is the key in the mocks dict to assert against.
    # expected_call_kwargs=None means assert_called_once() with no kwarg check (local).
    _DISPATCH_CASES: list[tuple[str, str, dict[str, Any] | None]] = [  # noqa: RUF012
        ("local", "local", None),
        ("docker", "docker", {"docker_executable": "docker", "image_overrides": None}),
        (
            "singularity",
            "singularity",
            {"singularity_executable": "singularity", "image_overrides": None},
        ),
        # apptainer delegates to use_singularity but passes its own executable name
        (
            "apptainer",
            "singularity",
            {"singularity_executable": "apptainer", "image_overrides": None},
        ),
    ]

    @pytest.mark.parametrize(
        ("runner_name", "mock_key", "expected_kwargs"),
        _DISPATCH_CASES,
        ids=[c[0] for c in _DISPATCH_CASES],
    )
    def test_runner_dispatch(
        self, runner_name: str, mock_key: str, expected_kwargs: dict[str, Any] | None
    ) -> None:
        """Test runner dispatch."""
        _, mocks = self._run(runner_name)
        if expected_kwargs is None:
            mocks[mock_key].assert_called_once()
        else:
            mocks[mock_key].assert_called_once_with(**expected_kwargs)

    def test_podman_constructs_podman_runner(self) -> None:
        """Podman is kept separate, asserts two mocks and uses a diff runner path."""
        _, mocks = self._run("podman")
        mocks["PodmanRunner"].assert_called_once_with(
            podman_executable="podman",
            image_overrides=None,
            podman_user_id=0,
        )
        mocks["set_global_runner"].assert_called_once()

    def test_unknown_runner_raises(self) -> None:
        """Assert raise if runner is incorrect."""
        import niwrap

        from niwrap_helper.niwrap import setup_styx

        with (
            patch.object(niwrap, "get_global_runner", return_value=_make_base_runner()),
            pytest.raises(NotImplementedError, match="Unknown runner"),
        ):
            setup_styx(runner="invalid")  # type: ignore[arg-type]

    # --- image_overrides forwarding ---

    def test_image_overrides_forwarded_to_docker(self) -> None:
        """Test image overrides work."""
        import niwrap

        from niwrap_helper.niwrap import setup_styx

        overrides = {"tool": "myimage:latest"}
        base = _make_base_runner()

        with (
            patch.object(niwrap, "get_global_runner", return_value=base),
            patch.object(niwrap, "use_docker") as mock_docker,
            patch("tempfile.mkdtemp", return_value="/tmp/x"),  # noqa: S108
        ):
            setup_styx(runner="docker", image_overrides=overrides)

        mock_docker.assert_called_once_with(
            docker_executable="docker",
            image_overrides=overrides,
        )

    # --- verbose / log level ---

    @pytest.mark.parametrize(
        ("verbose", "expected_level"),
        [
            (0, logging.WARNING),
            (1, logging.INFO),
            (2, logging.DEBUG),
            (99, logging.DEBUG),  # clamped to max index
        ],
    )
    def test_log_level(self, verbose: int, expected_level: int) -> None:
        """Test log level correctly set."""
        ctx, _ = self._run("local", verbose=verbose)
        assert ctx.logger.level == expected_level  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        ("verbose", "expected_verbose_flag"),
        [
            (0, False),
            (1, True),
            (2, True),
        ],
    )
    def test_verbose_flag(
        self,
        verbose: int,
        expected_verbose_flag: bool,  # noqa: FBT001
    ) -> None:
        """Test verbose flag is correctly set."""
        ctx, _ = self._run("local", verbose=verbose)
        assert ctx.verbose is expected_verbose_flag  # type: ignore[union-attr]

    # --- graph wrapping ---

    def test_graph_false_does_not_wrap(self) -> None:
        """Test no graph runner set."""
        _, mocks = self._run("local", graph=False)
        mocks["use_graph"].assert_not_called()

    def test_graph_true_wraps_runner(self) -> None:
        """Test graph runner set."""
        _, mocks = self._run("local", graph=True)
        mocks["use_graph"].assert_called_once()

    # --- return value ---

    def test_returns_styx_context(self) -> None:
        """Test NamedTuple (StyxContext) returned."""
        from niwrap_helper.niwrap import StyxContext

        ctx, _ = self._run("local")
        assert isinstance(ctx, StyxContext)

    def test_data_dir_set_to_tempdir(self) -> None:
        """Test correct setting of data directory."""
        import niwrap

        from niwrap_helper.niwrap import setup_styx

        base = _make_base_runner()
        with (
            patch.object(niwrap, "get_global_runner", return_value=base),
            patch.object(niwrap, "use_local"),
            patch("tempfile.mkdtemp", return_value="/tmp/styx_fake") as mock_mkdtemp,  # noqa: S108
        ):
            setup_styx(runner="local")

        mock_mkdtemp.assert_called_once_with(dir=None)
        assert base.data_dir == Path("/tmp/styx_fake")  # noqa: S108


# ---------------------------------------------------------------------------
# generate_exec_folder
# ---------------------------------------------------------------------------


class TestGenerateExecFolder:
    """Tests for generate_exec_folder."""

    def _setup(
        self,
        tmp_path: Path,
        execution_counter: int = 0,
    ) -> tuple[Path, MagicMock]:
        import niwrap

        from niwrap_helper.niwrap import generate_exec_folder

        base = _make_base_runner(
            data_dir=tmp_path,
            uid="uid1",
            execution_counter=execution_counter,
        )

        with patch.object(niwrap, "get_global_runner", return_value=base):
            result = generate_exec_folder()

        return result, base

    def test_returns_path(self, tmp_path: Path) -> None:
        """Test path is returned."""
        result, _ = self._setup(tmp_path)
        assert isinstance(result, Path)

    def test_folder_created_on_disk(self, tmp_path: Path) -> None:
        """Test directory created."""
        result, _ = self._setup(tmp_path)
        assert result.exists()
        assert result.is_dir()

    def test_folder_name_pattern(self, tmp_path: Path) -> None:
        """Test folder pattern is correct."""
        result, base = self._setup(tmp_path, execution_counter=0)
        assert result.name == f"{base.uid}_0_python"

    def test_custom_suffix(self, tmp_path: Path) -> None:
        """Test custom suffix used for folder."""
        import niwrap

        from niwrap_helper.niwrap import generate_exec_folder

        base = _make_base_runner(data_dir=tmp_path, uid="uid1", execution_counter=0)

        with patch.object(niwrap, "get_global_runner", return_value=base):
            result = generate_exec_folder(suffix="mytask")

        assert result.name.endswith("_mytask")

    def test_counter_incremented(self, tmp_path: Path) -> None:
        """Test counter incremented."""
        _, base = self._setup(tmp_path, execution_counter=3)
        assert base.execution_counter == 4

    def test_counter_increments_across_calls(self, tmp_path: Path) -> None:
        """Test counter incremented for runner state."""
        import niwrap

        from niwrap_helper.niwrap import generate_exec_folder

        base = _make_base_runner(data_dir=tmp_path, uid="uid1", execution_counter=0)

        with patch.object(niwrap, "get_global_runner", return_value=base):
            first = generate_exec_folder()
            second = generate_exec_folder()

        assert first.name == "uid1_0_python"
        assert second.name == "uid1_1_python"

    def test_graph_runner_unwrapped(self, tmp_path: Path) -> None:
        """Test graph runner is unwrapped to generate folder."""
        import niwrap

        from niwrap_helper.niwrap import generate_exec_folder

        base = _make_base_runner(data_dir=tmp_path, uid="uid1", execution_counter=0)
        graph = _make_graph_runner(base)

        with patch.object(niwrap, "get_global_runner", return_value=graph):
            result = generate_exec_folder()

        assert result.name == "uid1_0_python"


# ---------------------------------------------------------------------------
# cleanup_session
# ---------------------------------------------------------------------------


class TestCleanupSession:
    """Tests for cleanup_session."""

    def test_resets_execution_counter(self, tmp_path: Path) -> None:
        """Test counter reset."""
        import niwrap

        from niwrap_helper.niwrap import cleanup_session

        base = _make_base_runner(data_dir=tmp_path, execution_counter=7)

        with patch.object(niwrap, "get_global_runner", return_value=base):
            cleanup_session()

        assert base.execution_counter == 0

    def test_removes_data_dir(self, tmp_path: Path) -> None:
        """Test working directory removed."""
        import niwrap

        from niwrap_helper.niwrap import cleanup_session

        data_dir = tmp_path / "session_data"
        data_dir.mkdir()
        base = _make_base_runner(data_dir=data_dir)

        with patch.object(niwrap, "get_global_runner", return_value=base):
            cleanup_session()

        assert not data_dir.exists()

    def test_tolerates_missing_data_dir(self, tmp_path: Path) -> None:
        """cleanup_session must not raise if data_dir was already removed."""
        import niwrap

        from niwrap_helper.niwrap import cleanup_session

        base = _make_base_runner(data_dir=tmp_path / "nonexistent")

        with patch.object(niwrap, "get_global_runner", return_value=base):
            cleanup_session()  # must not raise

    def test_graph_runner_unwrapped(self, tmp_path: Path) -> None:
        """Test graph runner is unwrapped for cleanup."""
        import niwrap

        from niwrap_helper.niwrap import cleanup_session

        data_dir = tmp_path / "session_data"
        data_dir.mkdir()
        base = _make_base_runner(data_dir=data_dir, execution_counter=5)
        graph = _make_graph_runner(base)

        with patch.object(niwrap, "get_global_runner", return_value=graph):
            cleanup_session()

        assert base.execution_counter == 0
        assert not data_dir.exists()


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for save."""

    def test_single_path_copied(self, tmp_path: Path) -> None:
        """Test single file saved."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "src"
        src.mkdir()
        f = src / "result.nii"
        f.write_text("data")

        out = tmp_path / "out"
        save(f, out)

        assert (out / "result.nii").exists()

    def test_list_of_paths_all_copied(self, tmp_path: Path) -> None:
        """Test list of files saved."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "src"
        src.mkdir()
        files = [src / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("x")

        out = tmp_path / "out"
        save(files, out)

        for f in files:
            assert (out / f.name).exists()

    def test_out_dir_created_if_absent(self, tmp_path: Path) -> None:
        """Test output directory created if missing."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "f.txt"
        src.write_text("data")
        out = tmp_path / "deep" / "nested" / "out"

        save(src, out)

        assert out.is_dir()

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """Test strings accepted."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "f.txt"
        src.write_text("data")
        out = tmp_path / "out"

        save(str(src), out)  # type: ignore[arg-type]

        assert (out / "f.txt").exists()

    def test_empty_list_is_noop(self, tmp_path: Path) -> None:
        """Test nothing happens, but directory created if no files passed."""
        from niwrap_helper.niwrap import save

        out = tmp_path / "out"
        save([], out)  # must not raise
        assert out.is_dir()

    def test_existing_file_overwritten(self, tmp_path: Path) -> None:
        """Test existing files are overwritten."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "f.txt"
        src.write_text("new content")
        out = tmp_path / "out"
        out.mkdir()
        (out / "f.txt").write_text("old content")

        save(src, out)

        assert (out / "f.txt").read_text() == "new content"
