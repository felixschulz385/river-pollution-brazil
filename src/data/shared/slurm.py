"""Render and submit Slurm jobs equivalent to a `src.cli data` invocation.

Resource specs live in `setup/slurm_jobs.yaml`, keyed by `<source>.<verb>`
or `<source>.<verb>.<phase>` (see that file for the `defaults` block merged
into every entry). The job body is just the same CLI invocation the user
ran, minus `--slurm` -- there's a single source of truth for "what the job
runs."
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import yaml

_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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
    # `#SBATCH` lines are directives read by Slurm's own parser, not bash, but a
    # value containing a newline could still forge an extra directive; reject
    # that up front. Values used on lines bash actually executes (`eval`,
    # `conda activate`, `cd`, `mkdir`, `export`) are additionally
    # `shlex.quote`d below, since those are real shell-injection surfaces.
    for key in (
        "job_name",
        "partition",
        "time",
        "qos",
        "conda_hook",
        "conda_env",
        "project_dir",
        "cpus_per_task",
        "mem",
    ):
        value = spec.get(key)
        if isinstance(value, str) and "\n" in value:
            raise SlurmJobSpecError(f"Slurm job spec field {key!r} must not contain newlines.")
    if "\n" in str(log_dir):
        raise SlurmJobSpecError("log_dir must not contain newlines.")

    job_name = spec["job_name"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        # A single combined log, not a separate --error file: this repo's
        # CLI reports routine progress via `logging.info(...)` (see
        # `configure_logging()`), and Python's logging module defaults to
        # stderr for that -- so a split --output/--error would put nearly
        # everything worth reading in the "-error" file regardless of
        # whether it's actually an error, while --output stayed empty.
        f"#SBATCH --output={log_dir}/slurm-%j.log",
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
        f'eval "$({shlex.quote(spec["conda_hook"])} shell.bash hook)"',
        f"conda activate {shlex.quote(spec['conda_env'])}",
        "",
        f"cd {shlex.quote(spec['project_dir'])}",
        f"mkdir -p {shlex.quote(str(log_dir))}",
        "",
    ]
    for name, value in (spec.get("extra_env") or {}).items():
        if not _ENV_VAR_NAME_PATTERN.match(name):
            raise SlurmJobSpecError(f"Invalid extra_env variable name: {name!r}")
        lines.append(f"export {name}={shlex.quote(str(value))}")
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
