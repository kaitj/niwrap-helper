"""Tests for niwrap_helper.niwrap."""

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
    """Minimal mock resembling a StyxRunner.

    Uses spec=_FakeStyxRunner so isinstance(m, niwrap.GraphRunner) is False
    without fragile __class__ assignment.
    """
    m = MagicMock(spec=_FakeStyxRunner)
    m.data_dir = data_dir or Path("/tmp/styx_test")  # noqa: S108
    m.uid = uid
    m.execution_counter = execution_counter
    m.logger_name = logger_name
    return m


def _make_graph_runner(base: MagicMock) -> MagicMock:
    """Mock GraphRunner wrapping *base*."""
    import niwrap

    m = MagicMock(spec=niwrap.GraphRunner)
    m.base = base
    return m


# ---------------------------------------------------------------------------
# _get_base_runner
# ---------------------------------------------------------------------------


class TestGetBaseRunner:
    """Tests for _get_base_runner."""

    @pytest.mark.parametrize("use_graph", [False, True], ids=["plain", "graph"])
    def test_returns_base_runner(self, use_graph: bool) -> None:  # noqa: FBT001
        """Plain runner is returned as-is; GraphRunner is unwrapped to its base."""
        import niwrap

        from niwrap_helper.niwrap import _get_base_runner

        base = _make_base_runner()
        runner = _make_graph_runner(base) if use_graph else base

        with patch.object(niwrap, "get_global_runner", return_value=runner):
            assert _get_base_runner() is base


# ---------------------------------------------------------------------------
# setup_styx
# ---------------------------------------------------------------------------


class TestSetupStyx:
    """Tests for setup_styx: runner dispatch, logging, graph wrapping."""

    def _run(
        self,
        runner_name: str,
        verbose: int = 0,
        graph: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,  # noqa: ANN401
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
            ctx = setup_styx(runner=runner_name, verbose=verbose, graph=graph, **kwargs)  # type: ignore[arg-type]

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

    @pytest.mark.parametrize(
        ("runner_name", "mock_key", "expected_kwargs"),
        [
            ("local", "local", None),
            (
                "docker",
                "docker",
                {"docker_executable": "docker", "image_overrides": None},
            ),
            (
                "singularity",
                "singularity",
                {"singularity_executable": "singularity", "image_overrides": None},
            ),
            # apptainer delegates to use_singularity with its own executable name
            (
                "apptainer",
                "singularity",
                {"singularity_executable": "apptainer", "image_overrides": None},
            ),
        ],
    )
    def test_runner_dispatch(
        self, runner_name: str, mock_key: str, expected_kwargs: dict[str, Any] | None
    ) -> None:
        """Each runner name calls the correct backend with the right arguments."""
        _, mocks = self._run(runner_name)
        if expected_kwargs is None:
            mocks[mock_key].assert_called_once()
        else:
            mocks[mock_key].assert_called_once_with(**expected_kwargs)

    def test_podman_constructs_podman_runner(self) -> None:
        """Podman constructs PodmanRunner directly and calls set_global_runner."""
        _, mocks = self._run("podman")
        mocks["PodmanRunner"].assert_called_once_with(
            podman_executable="podman",
            image_overrides=None,
            podman_user_id=0,
        )
        mocks["set_global_runner"].assert_called_once()

    def test_unknown_runner_raises(self) -> None:
        """NotImplementedError raised for unrecognised runner name."""
        import niwrap

        from niwrap_helper.niwrap import setup_styx

        with (
            patch.object(niwrap, "get_global_runner", return_value=_make_base_runner()),
            pytest.raises(NotImplementedError, match="Unknown runner"),
        ):
            setup_styx(runner="invalid")  # type: ignore[arg-type]

    def test_image_overrides_forwarded(self) -> None:
        """image_overrides is passed through to the selected backend."""
        import niwrap

        from niwrap_helper.niwrap import setup_styx

        overrides = {"tool": "myimage:latest"}

        with (
            patch.object(niwrap, "get_global_runner", return_value=_make_base_runner()),
            patch.object(niwrap, "use_docker") as mock_docker,
            patch("tempfile.mkdtemp", return_value="/tmp/x"),  # noqa: S108
        ):
            setup_styx(runner="docker", image_overrides=overrides)

        mock_docker.assert_called_once_with(
            docker_executable="docker", image_overrides=overrides
        )

    # --- verbose / log level ---

    @pytest.mark.parametrize(
        ("verbose", "expected_level", "expected_verbose_flag"),
        [
            (0, logging.WARNING, False),
            (1, logging.INFO, True),
            (2, logging.DEBUG, True),
            (99, logging.DEBUG, True),  # clamped to max index
        ],
    )
    def test_verbosity(
        self,
        verbose: int,
        expected_level: int,
        expected_verbose_flag: bool,  # noqa: FBT001
    ) -> None:
        """Logger level and verbose flag are set correctly for each verbosity value."""
        ctx, _ = self._run("local", verbose=verbose)
        assert ctx.logger.level == expected_level  # type: ignore[union-attr]
        assert ctx.verbose is expected_verbose_flag  # type: ignore[union-attr]

    # --- graph wrapping ---

    @pytest.mark.parametrize("graph", [False, True], ids=["no-graph", "graph"])
    def test_graph_wrapping(self, graph: bool) -> None:  # noqa: FBT001
        """use_graph is called iff graph=True."""
        _, mocks = self._run("local", graph=graph)
        if graph:
            mocks["use_graph"].assert_called_once()
        else:
            mocks["use_graph"].assert_not_called()

    # --- return value / data_dir ---

    def test_returns_styx_context_with_tempdir(self) -> None:
        """Returns a StyxContext and sets base.data_dir to the mkdtemp result."""
        import niwrap

        from niwrap_helper.niwrap import StyxContext, setup_styx

        base = _make_base_runner()

        with (
            patch.object(niwrap, "get_global_runner", return_value=base),
            patch.object(niwrap, "use_local"),
            patch("tempfile.mkdtemp", return_value="/tmp/styx_fake") as mock_mkdtemp,  # noqa: S108
        ):
            ctx = setup_styx(runner="local")

        assert isinstance(ctx, StyxContext)
        mock_mkdtemp.assert_called_once_with(dir=None)
        assert base.data_dir == Path("/tmp/styx_fake")  # noqa: S108


# ---------------------------------------------------------------------------
# generate_exec_folder
# ---------------------------------------------------------------------------


class TestGenerateExecFolder:
    """Tests for generate_exec_folder."""

    def _call(
        self,
        tmp_path: Path,
        uid: str = "uid1",
        execution_counter: int = 0,
        suffix: str | None = None,
        use_graph: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[Path, MagicMock]:
        import niwrap

        from niwrap_helper.niwrap import generate_exec_folder

        base = _make_base_runner(
            data_dir=tmp_path, uid=uid, execution_counter=execution_counter
        )
        runner = _make_graph_runner(base) if use_graph else base

        kwargs = {"suffix": suffix} if suffix is not None else {}
        with patch.object(niwrap, "get_global_runner", return_value=runner):
            result = generate_exec_folder(**kwargs)

        return result, base

    def test_creates_dir_with_correct_name(self, tmp_path: Path) -> None:
        """Creates a directory named {uid}_{counter}_python on disk."""
        result, base = self._call(tmp_path)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()
        assert result.name == f"{base.uid}_0_python"

    def test_custom_suffix(self, tmp_path: Path) -> None:
        """Custom suffix replaces the default 'python' suffix in the folder name."""
        result, _ = self._call(tmp_path, suffix="mytask")
        assert result.name.endswith("_mytask")

    def test_counter_incremented(self, tmp_path: Path) -> None:
        """Execution counter is incremented by 1 after each call."""
        _, base = self._call(tmp_path, execution_counter=3)
        assert base.execution_counter == 4

    def test_counter_increments_across_calls(self, tmp_path: Path) -> None:
        """Sequential calls produce unique, incrementing folder names."""
        import niwrap

        from niwrap_helper.niwrap import generate_exec_folder

        base = _make_base_runner(data_dir=tmp_path, uid="uid1", execution_counter=0)

        with patch.object(niwrap, "get_global_runner", return_value=base):
            first = generate_exec_folder()
            second = generate_exec_folder()

        assert first.name == "uid1_0_python"
        assert second.name == "uid1_1_python"

    def test_graph_runner_unwrapped(self, tmp_path: Path) -> None:
        """GraphRunner is unwrapped before generating the folder."""
        result, _ = self._call(tmp_path, use_graph=True)
        assert result.name == "uid1_0_python"


# ---------------------------------------------------------------------------
# cleanup_session
# ---------------------------------------------------------------------------


class TestCleanupSession:
    """Tests for cleanup_session."""

    def _call(
        self,
        tmp_path: Path,
        data_dir: Path | None = None,
        execution_counter: int = 0,
        use_graph: bool = False,  # noqa: FBT001, FBT002
    ) -> MagicMock:
        import niwrap

        from niwrap_helper.niwrap import cleanup_session

        base = _make_base_runner(
            data_dir=data_dir or tmp_path, execution_counter=execution_counter
        )
        runner = _make_graph_runner(base) if use_graph else base

        with patch.object(niwrap, "get_global_runner", return_value=runner):
            cleanup_session()

        return base

    def test_resets_counter_and_removes_data_dir(self, tmp_path: Path) -> None:
        """Counter is reset to 0 and data_dir is removed from disk."""
        data_dir = tmp_path / "session_data"
        data_dir.mkdir()
        base = self._call(tmp_path, data_dir=data_dir, execution_counter=7)
        assert base.execution_counter == 0
        assert not data_dir.exists()

    def test_tolerates_missing_data_dir(self, tmp_path: Path) -> None:
        """Does not raise if data_dir was already removed."""
        self._call(tmp_path, data_dir=tmp_path / "nonexistent")  # must not raise

    def test_graph_runner_unwrapped(self, tmp_path: Path) -> None:
        """GraphRunner is unwrapped before resetting counter and removing data_dir."""
        data_dir = tmp_path / "session_data"
        data_dir.mkdir()
        base = self._call(
            tmp_path, data_dir=data_dir, execution_counter=5, use_graph=True
        )
        assert base.execution_counter == 0
        assert not data_dir.exists()


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    """Tests for save."""

    @pytest.mark.parametrize(
        ("inputs", "expected_names"),
        [
            ("single", ["result.nii"]),
            ("list", ["file_0.txt", "file_1.txt", "file_2.txt"]),
            ("string", ["f.txt"]),
            ("empty", []),
        ],
        ids=["single-path", "list-of-paths", "string-path", "empty-list"],
    )
    def test_copies_inputs_to_output_dir(
        self, tmp_path: Path, inputs: str, expected_names: list[str]
    ) -> None:
        """Files are copied to output dir; empty list and string paths are accepted."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"

        if inputs == "single":
            f = src / "result.nii"
            f.write_text("data")
            save(f, out)
        elif inputs == "list":
            files = [src / name for name in expected_names]
            for f in files:
                f.write_text("x")
            save(files, out)
        elif inputs == "string":
            f = src / "f.txt"
            f.write_text("data")
            save(str(f), out)  # type: ignore[arg-type]
        else:
            save([], out)

        assert out.is_dir()
        for name in expected_names:
            assert (out / name).exists()

    def test_creates_nested_output_dir(self, tmp_path: Path) -> None:
        """Output directory is created recursively if it does not exist."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "f.txt"
        src.write_text("data")
        out = tmp_path / "deep" / "nested" / "out"
        save(src, out)
        assert out.is_dir()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Existing files in the output directory are overwritten."""
        from niwrap_helper.niwrap import save

        src = tmp_path / "f.txt"
        src.write_text("new content")
        out = tmp_path / "out"
        out.mkdir()
        (out / "f.txt").write_text("old content")
        save(src, out)
        assert (out / "f.txt").read_text() == "new content"
