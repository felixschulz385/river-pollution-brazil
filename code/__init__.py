"""Project package with compatibility shims for the stdlib ``code`` module.

This repository's top-level package is named ``code``, which shadows Python's
standard-library module of the same name. Some third-party dependencies import
``code.InteractiveConsole`` from the stdlib, so we re-export that API here while
preserving package-style imports such as ``code.analysis``.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sysconfig


def _load_stdlib_code_module():
    stdlib_code_path = Path(sysconfig.get_paths()["stdlib"]) / "code.py"
    spec = spec_from_file_location("_stdlib_code", stdlib_code_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Could not load stdlib code module from {stdlib_code_path}.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_code = _load_stdlib_code_module()

InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
InteractiveConsole = _stdlib_code.InteractiveConsole
interact = _stdlib_code.interact
compile_command = _stdlib_code.compile_command

__all__ = [
    "InteractiveInterpreter",
    "InteractiveConsole",
    "interact",
    "compile_command",
]
