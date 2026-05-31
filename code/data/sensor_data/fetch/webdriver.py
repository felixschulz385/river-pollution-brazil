"""Compatibility wrapper for shared Chrome WebDriver lifecycle management."""

from shared.webdriver import ManagedBrowser, open_browser

__all__ = ["ManagedBrowser", "open_browser"]
