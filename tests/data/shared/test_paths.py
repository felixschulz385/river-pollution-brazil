from __future__ import annotations

from src.data.shared.paths import SCRATCH_DIR_ENV_VAR, SCRATCH_DIR_NAME, scratch_root


def test_scratch_root_defaults_under_project_root(tmp_path, monkeypatch):
    monkeypatch.delenv(SCRATCH_DIR_ENV_VAR, raising=False)

    resolved = scratch_root(tmp_path)

    assert resolved == (tmp_path / SCRATCH_DIR_NAME).resolve()
    assert resolved.is_absolute()
    assert resolved.is_dir()


def test_scratch_root_env_var_overrides_and_is_created(tmp_path, monkeypatch):
    override = tmp_path / "node_local" / "tmp"
    monkeypatch.setenv(SCRATCH_DIR_ENV_VAR, str(override))

    resolved = scratch_root("/some/project/root")

    assert resolved == override.resolve()
    assert resolved.is_dir()
