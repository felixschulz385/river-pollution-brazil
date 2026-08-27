import subprocess
from pathlib import Path

import pytest
import yaml

from src.data.shared.slurm import (
    SlurmJobSpecError,
    load_job_spec,
    render_sbatch_script,
    resolve_n_jobs,
    submit_job,
)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "slurm_jobs.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "partition": "scicore",
                    "conda_hook": "/opt/conda/bin/conda",
                    "conda_env": "311",
                    "project_dir": "/project",
                },
                "jobs": {
                    "climate.preprocess.extract": {
                        "job_name": "extract_climate",
                        "time": "0-12:00:00",
                        "qos": "1day",
                        "cpus_per_task": 4,
                        "mem": "128G",
                        "extra_env": {"OMP_NUM_THREADS": 1},
                    }
                },
            }
        )
    )
    return path


def test_load_job_spec_merges_defaults(config_path):
    spec = load_job_spec("climate.preprocess.extract", config_path=config_path)
    assert spec["job_name"] == "extract_climate"
    assert spec["partition"] == "scicore"
    assert spec["conda_env"] == "311"


def test_load_job_spec_missing_key_raises_with_available_keys(config_path):
    with pytest.raises(SlurmJobSpecError, match="climate.preprocess.extract"):
        load_job_spec("does.not.exist", config_path=config_path)


def test_render_sbatch_script_includes_directives_and_command(config_path):
    spec = load_job_spec("climate.preprocess.extract", config_path=config_path)
    script = render_sbatch_script(
        spec,
        ["data", "preprocess", "--source", "climate", "--phase", "extract"],
        log_dir="log/climate_preprocess",
    )
    assert "#SBATCH --job-name=extract_climate" in script
    assert "#SBATCH --partition=scicore" in script
    assert "#SBATCH --time=0-12:00:00" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=128G" in script
    assert "conda activate 311" in script
    assert "cd /project" in script
    assert "export OMP_NUM_THREADS=1" in script
    assert "python -m src.cli data preprocess --source climate --phase extract" in script


def test_render_sbatch_script_quotes_shell_metacharacters_in_spec_values(config_path):
    spec = load_job_spec("climate.preprocess.extract", config_path=config_path)
    spec["conda_env"] = "311; rm -rf /"
    spec["project_dir"] = "/project $(whoami)"

    script = render_sbatch_script(spec, ["data", "preprocess"], log_dir="log/climate")

    assert "conda activate '311; rm -rf /'" in script
    assert "cd '/project $(whoami)'" in script


def test_render_sbatch_script_rejects_invalid_extra_env_name(config_path):
    spec = load_job_spec("climate.preprocess.extract", config_path=config_path)
    spec["extra_env"] = {"NOT VALID; rm -rf /": "1"}

    with pytest.raises(SlurmJobSpecError):
        render_sbatch_script(spec, ["data", "preprocess"], log_dir="log/climate")


def test_render_sbatch_script_rejects_newline_in_job_name(config_path):
    spec = load_job_spec("climate.preprocess.extract", config_path=config_path)
    spec["job_name"] = "extract\n#SBATCH --partition=forged"

    with pytest.raises(SlurmJobSpecError):
        render_sbatch_script(spec, ["data", "preprocess"], log_dir="log/climate")


def test_render_sbatch_script_rejects_newline_in_mem(config_path):
    spec = load_job_spec("climate.preprocess.extract", config_path=config_path)
    spec["mem"] = "128G\n#SBATCH --partition=forged"

    with pytest.raises(SlurmJobSpecError):
        render_sbatch_script(spec, ["data", "preprocess"], log_dir="log/climate")


def test_submit_job_parses_job_id(monkeypatch):
    def fake_run(cmd, input, text, capture_output, check):
        assert cmd == ["sbatch"]
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 12345\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert submit_job("#!/bin/bash\necho hi\n") == "12345"


def test_submit_job_raises_on_unparseable_output(monkeypatch):
    def fake_run(cmd, input, text, capture_output, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="something unexpected\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SlurmJobSpecError):
        submit_job("#!/bin/bash\necho hi\n")


def test_resolve_n_jobs_uses_slurm_allocation_when_present(monkeypatch):
    """Regression test for the OOM this fixes: land_cover preprocessing used
    to default to os.cpu_count() (the full node's core count, e.g. 128),
    which vastly oversubscribed the job's actual Slurm memory allocation."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert resolve_n_jobs() == 8


def test_resolve_n_jobs_falls_back_to_cpu_count_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert resolve_n_jobs() == 4


def test_resolve_n_jobs_ignores_unparseable_slurm_value(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert resolve_n_jobs() == 4


def test_resolve_n_jobs_never_returns_less_than_one(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "0")
    assert resolve_n_jobs() == 1
