"""Render and submit Slurm jobs equivalent to a `src.cli data` invocation.

Resource specs live in `setup/slurm_jobs.yaml`, keyed by `<source>.<verb>`
or `<source>.<verb>.<phase>` (see that file for the `defaults` block merged
into every entry). The job body is just the same CLI invocation the user
ran, minus `--slurm` -- there's a single source of truth for "what the job
runs."
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import yaml

DEFAULT_JOB_CONFIG_PATH = Path(__file__).resolve().parents[3] / "setup" / "slurm_jobs.yaml"


class SlurmJobSpecError(Exception):
    """Raised when `setup/slurm_jobs.yaml` has no entry for a requested key."""


def load_job_spec(key: str, config_path: Path = DEFAULT_JOB_CONFIG_PATH) -> dict:
    """Return the resource spec for `key`, merged with the config's `defaults` block."""
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    jobs = config.get("jobs", {})
    if key not in jobs:
        available = ", ".join(sorted(jobs)) or "(none defined)"
        raise SlurmJobSpecError(
            f"No Slurm job defined for '{key}' in {config_path}. "
            f"Available keys: {available}. Add an entry or run without --slurm."
        )
    return {**config.get("defaults", {}), **jobs[key]}


def render_sbatch_script(spec: dict, command_argv: Sequence[str], log_dir: str) -> str:
    """Render an sbatch script that runs `python -m src.cli <command_argv>`."""
    job_name = spec["job_name"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/slurm-%j.log",
        f"#SBATCH --error={log_dir}/slurm-%j-error.log",
        f"#SBATCH --partition={spec['partition']}",
        f"#SBATCH --time={spec['time']}",
    ]
    if spec.get("qos"):
        lines.append(f"#SBATCH --qos={spec['qos']}")
    lines.append(f"#SBATCH --cpus-per-task={spec['cpus_per_task']}")
    lines.append(f"#SBATCH --mem={spec['mem']}")
    lines += [
        "",
        "set -euo pipefail",
        "",
        f'eval "$({spec["conda_hook"]} shell.bash hook)"',
        f"conda activate {spec['conda_env']}",
        "",
        f"cd {spec['project_dir']}",
        f"mkdir -p {log_dir}",
        "",
    ]
    for name, value in (spec.get("extra_env") or {}).items():
        lines.append(f"export {name}={value}")
    if spec.get("extra_env"):
        lines.append("")

    command = "python -m src.cli " + " ".join(shlex.quote(arg) for arg in command_argv)
    lines.append(command)
    lines.append("")
    return "\n".join(lines)


def submit_job(script_text: str) -> str:
    """Submit `script_text` via `sbatch` and return the new job id."""
    result = subprocess.run(
        ["sbatch"],
        input=script_text,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = result.stdout.strip()
    # Slurm prints "Submitted batch job <id>".
    job_id = stdout.rsplit(" ", 1)[-1]
    if not job_id.isdigit():
        raise SlurmJobSpecError(f"Could not parse job id from sbatch output: {stdout!r}")
    return job_id
