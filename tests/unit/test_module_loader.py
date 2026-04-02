# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared module loader utility."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from arksim.utils.module_loader import _module_cache, load_module_from_file

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_module_cache() -> None:
    """Clear the module cache between tests."""
    _module_cache.clear()


@pytest.fixture
def valid_module(tmp_path: Path) -> Path:
    """Create a valid .py module with a simple class."""
    code = textwrap.dedent("""\
        class HelloWorld:
            greeting = "hello"
    """)
    module_file = tmp_path / "valid_mod.py"
    module_file.write_text(code)
    return module_file


@pytest.fixture
def broken_module(tmp_path: Path) -> Path:
    """Create a .py module that raises on import."""
    module_file = tmp_path / "broken_mod.py"
    module_file.write_text("raise RuntimeError('intentional failure')")
    return module_file


@pytest.fixture
def sibling_import_module(tmp_path: Path) -> Path:
    """Create a module that imports from a sibling package."""
    # Create sibling package
    pkg_dir = tmp_path / "sibling_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "helper.py").write_text("VALUE = 42\n")

    # Create module that imports from sibling
    code = textwrap.dedent("""\
        from sibling_pkg.helper import VALUE

        class MyClass:
            value = VALUE
    """)
    module_file = tmp_path / "importer_mod.py"
    module_file.write_text(code)
    return module_file


# ── Happy path tests ───────────────────────────────────────────────────────


class TestLoadModuleFromFile:
    """Tests for successful module loading."""

    def test_loads_valid_module(self, valid_module: Path) -> None:
        module = load_module_from_file(str(valid_module))

        assert hasattr(module, "HelloWorld")
        assert module.HelloWorld.greeting == "hello"

    def test_cached_by_file_path(self, valid_module: Path) -> None:
        """Same file path returns the same cached module."""
        mod1 = load_module_from_file(str(valid_module))
        mod2 = load_module_from_file(str(valid_module))

        assert mod1 is mod2
        assert mod1.__name__ == mod2.__name__
        assert mod1.__name__.startswith("_arksim_")

    def test_module_registered_in_sys_modules(self, valid_module: Path) -> None:
        module = load_module_from_file(str(valid_module))

        assert module.__name__ in sys.modules
        assert sys.modules[module.__name__] is module

    def test_parent_dir_added_to_sys_path(self, valid_module: Path) -> None:
        parent_dir = str(valid_module.parent.resolve())
        # Remove if already present (clean state)
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)

        load_module_from_file(str(valid_module))

        assert parent_dir in sys.path

    def test_sibling_package_import(self, sibling_import_module: Path) -> None:
        """Module can import from sibling packages via sys.path."""
        module = load_module_from_file(str(sibling_import_module))

        assert hasattr(module, "MyClass")
        assert module.MyClass.value == 42

    def test_accepts_absolute_path(self, valid_module: Path) -> None:
        abs_path = str(valid_module.resolve())
        module = load_module_from_file(abs_path)

        assert hasattr(module, "HelloWorld")


# ── Error handling tests ───────────────────────────────────────────────────


class TestLoadModuleErrors:
    """Tests for error handling in module loading."""

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Module file not found"):
            load_module_from_file("/nonexistent/path/module.py")

    def test_non_py_file_raises_value_error(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "module.txt"
        txt_file.write_text("not a python file")

        with pytest.raises(ValueError, match="must be a .py file"):
            load_module_from_file(str(txt_file))

    def test_broken_module_raises_runtime_error(self, broken_module: Path) -> None:
        with pytest.raises(RuntimeError, match="Failed to load module from"):
            load_module_from_file(str(broken_module))

    def test_broken_module_cleans_up_sys_modules(self, broken_module: Path) -> None:
        """On failure, the module entry is removed from sys.modules."""
        arksim_modules_before = {k for k in sys.modules if k.startswith("_arksim_")}

        with pytest.raises(RuntimeError):
            load_module_from_file(str(broken_module))

        arksim_modules_after = {k for k in sys.modules if k.startswith("_arksim_")}
        # No new _arksim_ entries should remain
        assert arksim_modules_after == arksim_modules_before
