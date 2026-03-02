"""Tests for output utilities."""

import json
import os

import pytest

from arksim.utils.output import load_json_file, save_json_file


class TestLoadJsonFile:
    """Tests for load_json_file function."""

    def test_loads_valid_json(self, temp_dir: dict) -> None:
        """Test loading a valid JSON file."""
        file_path = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 42}
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_json_file(file_path)

        assert result == data

    def test_loads_nested_json(self, temp_dir: dict) -> None:
        """Test loading nested JSON structure."""
        file_path = os.path.join(temp_dir, "nested.json")
        data = {"outer": {"inner": [1, 2, 3]}, "list": ["a", "b"]}
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_json_file(file_path)

        assert result == data

    def test_raises_for_nonexistent_file(self) -> None:
        """Test raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_json_file("/nonexistent/path.json")

    def test_raises_for_invalid_json(self, temp_dir: dict) -> None:
        """Test raises JSONDecodeError for invalid JSON."""
        file_path = os.path.join(temp_dir, "invalid.json")
        with open(file_path, "w") as f:
            f.write("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_json_file(file_path)

    def test_loads_empty_object(self, temp_dir: dict) -> None:
        """Test loading empty JSON object."""
        file_path = os.path.join(temp_dir, "empty.json")
        with open(file_path, "w") as f:
            f.write("{}")

        result = load_json_file(file_path)

        assert result == {}

    def test_loads_json_array(self, temp_dir: dict) -> None:
        """Test loading JSON array."""
        file_path = os.path.join(temp_dir, "array.json")
        data = [1, 2, 3, "four"]
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_json_file(file_path)

        assert result == data


class TestSaveJsonFile:
    """Tests for save_json_file function."""

    def test_saves_valid_json(self, temp_dir: dict) -> None:
        """Test saving a valid JSON file."""
        file_path = os.path.join(temp_dir, "output.json")
        data = {"key": "value", "number": 42}

        save_json_file(data, file_path)

        with open(file_path) as f:
            result = json.load(f)
        assert result == data

    def test_saves_with_custom_indent(self, temp_dir: dict) -> None:
        """Test saving JSON with custom indentation."""
        file_path = os.path.join(temp_dir, "indented.json")
        data = {"key": "value"}

        save_json_file(data, file_path, indent=2)

        with open(file_path) as f:
            content = f.read()
        # Check indentation is 2 spaces
        assert "  " in content

    def test_raises_when_file_exists_without_overwrite(self, temp_dir: dict) -> None:
        """Test raises FileExistsError when file exists and overwrite=False."""
        file_path = os.path.join(temp_dir, "existing.json")
        with open(file_path, "w") as f:
            f.write("{}")

        with pytest.raises(FileExistsError, match="Set overwrite=True"):
            save_json_file({"new": "data"}, file_path)

    def test_overwrites_when_overwrite_true(self, temp_dir: dict) -> None:
        """Test overwrites file when overwrite=True."""
        file_path = os.path.join(temp_dir, "overwrite.json")
        with open(file_path, "w") as f:
            json.dump({"old": "data"}, f)

        save_json_file({"new": "data"}, file_path, overwrite=True)

        with open(file_path) as f:
            result = json.load(f)
        assert result == {"new": "data"}

    def test_creates_new_file(self, temp_dir: dict) -> None:
        """Test creates new file when it doesn't exist."""
        file_path = os.path.join(temp_dir, "new_file.json")
        data = {"created": True}

        save_json_file(data, file_path)

        assert os.path.exists(file_path)
