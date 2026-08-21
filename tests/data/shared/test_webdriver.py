from __future__ import annotations

from pathlib import Path

from src.data.shared.webdriver import ManagedBrowser


def test_managed_browser_exit_skips_quit_and_warns_when_keep_open_on_error(
    monkeypatch, caplog, tmp_path: Path
):
    browser = ManagedBrowser(keep_open_on_error=True)
    browser._profile_dir = tmp_path / "profile"

    quit_calls = []
    monkeypatch.setattr(browser, "quit", lambda: quit_calls.append(True))

    with caplog.at_level("WARNING", logger="src.data.shared.webdriver"):
        result = browser.__exit__(RuntimeError, RuntimeError("boom"), None)

    assert result is False
    assert quit_calls == []
    assert any("will NOT be cleaned up" in message for message in caplog.messages)


def test_managed_browser_exit_calls_quit_without_keep_open_on_error(monkeypatch):
    browser = ManagedBrowser(keep_open_on_error=False)

    quit_calls = []
    monkeypatch.setattr(browser, "quit", lambda: quit_calls.append(True))

    result = browser.__exit__(RuntimeError, RuntimeError("boom"), None)

    assert result is False
    assert quit_calls == [True]
