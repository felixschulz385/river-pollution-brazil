from __future__ import annotations

import time

from src.data.verification.fingerprint import compute_fingerprint


def test_fingerprint_stable_when_files_unchanged(tmp_path):
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")
    file_b = tmp_path / "b.txt"
    file_b.write_text("world")

    first = compute_fingerprint([file_a, file_b])
    second = compute_fingerprint([file_a, file_b])

    assert first == second


def test_fingerprint_changes_on_size_change(tmp_path):
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")

    before = compute_fingerprint([file_a])
    file_a.write_text("hello world, this is longer")
    after = compute_fingerprint([file_a])

    assert before != after


def test_fingerprint_changes_on_mtime_change(tmp_path):
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")

    before = compute_fingerprint([file_a])
    # Bump mtime without changing size or content.
    new_time = time.time() + 10
    import os

    os.utime(file_a, (new_time, new_time))
    after = compute_fingerprint([file_a])

    assert before != after


def test_fingerprint_handles_missing_files_without_raising(tmp_path):
    missing = tmp_path / "does_not_exist.txt"

    fingerprint = compute_fingerprint([missing])

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64  # sha256 hex digest


def test_fingerprint_differs_between_present_and_missing(tmp_path):
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")
    missing = tmp_path / "missing.txt"

    present_fp = compute_fingerprint([file_a])
    file_a.unlink()
    missing_fp = compute_fingerprint([file_a])

    assert present_fp != missing_fp
    # Sanity: fingerprinting an always-missing path is stable too.
    assert compute_fingerprint([missing]) == compute_fingerprint([missing])
