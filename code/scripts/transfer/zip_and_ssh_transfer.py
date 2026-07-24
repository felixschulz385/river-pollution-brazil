#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zip a processed dataset and transfer it to a remote directory over SSH."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the processed dataset file or directory to archive.",
    )
    parser.add_argument(
        "remote",
        help="SSH target in the form user@host.",
    )
    parser.add_argument(
        "remote_dir",
        help="Remote directory that should receive the zip archive.",
    )
    parser.add_argument(
        "--archive-name",
        default=None,
        help="Optional archive filename. Defaults to <dataset-name>_<UTC timestamp>.zip.",
    )
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Optional local directory for the temporary zip file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional SSH port for both ssh and scp.",
    )
    parser.add_argument(
        "--identity-file",
        default=None,
        help="Optional path to an SSH private key file passed to ssh/scp with -i.",
    )
    parser.add_argument(
        "--ssh-binary",
        default="ssh",
        help="SSH executable to use (default: ssh).",
    )
    parser.add_argument(
        "--scp-binary",
        default="scp",
        help="SCP executable to use (default: scp).",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the created zip archive instead of deleting it after upload.",
    )
    return parser.parse_args()


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find required executable '{name}' on PATH."
        )
    return resolved


def normalize_dataset_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return path


def normalize_optional_file_path(raw_path: str | None, *, label: str) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def build_archive_path(
    dataset_path: Path,
    archive_name: str | None,
    staging_dir: str | None,
) -> Path:
    target_dir = (
        Path(staging_dir).expanduser().resolve()
        if staging_dir
        else Path(tempfile.gettempdir()).resolve()
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    if archive_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_name = f"{dataset_path.name}_{timestamp}.zip"
    elif not archive_name.endswith(".zip"):
        archive_name = f"{archive_name}.zip"

    return target_dir / archive_name


def zip_dataset(dataset_path: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if dataset_path.is_file():
            zf.write(dataset_path, arcname=dataset_path.name)
            return

        root_parent = dataset_path.parent
        for child in sorted(dataset_path.rglob("*")):
            if child.is_dir():
                continue
            zf.write(child, arcname=child.relative_to(root_parent))


def remote_archive_path(remote_dir: str, archive_name: str) -> str:
    trimmed = remote_dir.rstrip("/")
    if not trimmed:
        return archive_name
    return f"{trimmed}/{archive_name}"


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _windows_current_user() -> str:
    return subprocess.run(
        ["whoami"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _lock_down_windows_private_key(path: Path) -> None:
    current_user = _windows_current_user()
    subprocess.run(
        ["icacls", str(path), "/inheritance:r"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["icacls", str(path), "/grant:r", f"{current_user}:R"],
        check=True,
        capture_output=True,
        text=True,
    )


def stage_identity_file_for_ssh(identity_file: Path | None) -> tuple[Path | None, Path | None]:
    if identity_file is None:
        return None, None

    if os.name != "nt":
        return identity_file, None

    temp_dir = Path(tempfile.mkdtemp(prefix="ssh_key_stage_")).resolve()
    staged_path = temp_dir / identity_file.name
    shutil.copy2(identity_file, staged_path)
    _lock_down_windows_private_key(staged_path)
    return staged_path, temp_dir


def ensure_remote_dir(
    ssh_binary: str,
    remote: str,
    remote_dir: str,
    port: int | None,
    identity_file: Path | None,
) -> None:
    cmd = [ssh_binary]
    if port is not None:
        cmd.extend(["-p", str(port)])
    if identity_file is not None:
        cmd.extend(["-i", str(identity_file)])
    cmd.extend([remote, f"mkdir -p {shlex.quote(remote_dir)}"])
    run_command(cmd)


def copy_archive(
    scp_binary: str,
    archive_path: Path,
    remote: str,
    remote_dir: str,
    port: int | None,
    identity_file: Path | None,
) -> str:
    destination_path = remote_archive_path(remote_dir, archive_path.name)
    quoted_destination = shlex.quote(destination_path)

    cmd = [scp_binary]
    if port is not None:
        cmd.extend(["-P", str(port)])
    if identity_file is not None:
        cmd.extend(["-i", str(identity_file)])
    cmd.extend([str(archive_path), f"{remote}:{quoted_destination}"])
    run_command(cmd)
    return destination_path


def main() -> int:
    args = parse_args()
    staged_identity_dir: Path | None = None

    try:
        ssh_binary = require_binary(args.ssh_binary)
        scp_binary = require_binary(args.scp_binary)
        dataset_path = normalize_dataset_path(args.dataset_path)
        identity_file = normalize_optional_file_path(
            args.identity_file,
            label="Identity file",
        )
        ssh_identity_file, staged_identity_dir = stage_identity_file_for_ssh(identity_file)
        archive_path = build_archive_path(
            dataset_path=dataset_path,
            archive_name=args.archive_name,
            staging_dir=args.staging_dir,
        )

        print(f"Creating archive: {archive_path}")
        zip_dataset(dataset_path, archive_path)

        print(f"Ensuring remote directory exists: {args.remote}:{args.remote_dir}")
        ensure_remote_dir(
            ssh_binary=ssh_binary,
            remote=args.remote,
            remote_dir=args.remote_dir,
            port=args.port,
            identity_file=ssh_identity_file,
        )

        print(f"Uploading archive to {args.remote}")
        uploaded_path = copy_archive(
            scp_binary=scp_binary,
            archive_path=archive_path,
            remote=args.remote,
            remote_dir=args.remote_dir,
            port=args.port,
            identity_file=ssh_identity_file,
        )
        print(f"Upload complete: {args.remote}:{uploaded_path}")

        if args.keep_archive:
            print(f"Kept local archive: {archive_path}")
        else:
            archive_path.unlink(missing_ok=True)
            print("Deleted local archive after transfer.")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"Transfer failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if staged_identity_dir is not None:
            shutil.rmtree(staged_identity_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
