# SPDX-License-Identifier: Apache-2.0
"""Tests for filesystem API endpoints via FastAPI TestClient."""

from __future__ import annotations

import json
import os

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arksim.ui.api.routes_filesystem import router


@pytest.fixture()
def fs_app(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    root = str(tmp_path)
    monkeypatch.setattr("arksim.ui.api.routes_filesystem.PROJECT_ROOT", root)
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def fs_client(fs_app: FastAPI) -> TestClient:
    return TestClient(fs_app)


@pytest.fixture()
def fs_root(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    root = str(tmp_path)
    monkeypatch.setattr("arksim.ui.api.routes_filesystem.PROJECT_ROOT", root)
    return root


class TestLoadConfig:
    def test_valid_yaml_file(self, fs_client: TestClient, fs_root: str) -> None:
        cfg_path = os.path.join(fs_root, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump({"key": "value"}, f)
        resp = fs_client.get("/fs/config", params={"path": cfg_path})
        assert resp.status_code == 200
        assert resp.json()["settings"]["key"] == "value"

    def test_traversal_attempt(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.get("/fs/config", params={"path": "../../etc/passwd"})
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert "outside project" in resp.json()["error"].lower()

    def test_nonexistent_file(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.get("/fs/config", params={"path": "missing.yaml"})
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert "not found" in resp.json()["error"].lower()

    def test_resolves_relative_path_keys(
        self, fs_client: TestClient, fs_root: str
    ) -> None:
        cfg_data = {
            "agent_config_file_path": "./agents/agent.json",
            "scenario_file_path": "scenarios/s.json",
        }
        cfg_path = os.path.join(fs_root, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_data, f)
        resp = fs_client.get("/fs/config", params={"path": cfg_path})
        settings = resp.json()["settings"]
        assert settings["agent_config_file_path"] == os.path.join(
            fs_root, "agents", "agent.json"
        )
        assert settings["scenario_file_path"] == os.path.join(
            fs_root, "scenarios", "s.json"
        )

    def test_resolves_list_path_keys(self, fs_client: TestClient, fs_root: str) -> None:
        cfg_data = {
            "custom_metrics_file_paths": ["./metrics/m1.py", "./metrics/m2.py"],
        }
        cfg_path = os.path.join(fs_root, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_data, f)
        resp = fs_client.get("/fs/config", params={"path": cfg_path})
        settings = resp.json()["settings"]
        assert len(settings["custom_metrics_file_paths"]) == 2


class TestLoadScenario:
    def test_valid_json(self, fs_client: TestClient, fs_root: str) -> None:
        scenario_path = os.path.join(fs_root, "scenario.json")
        data = {"conversations": []}
        with open(scenario_path, "w") as f:
            json.dump(data, f)
        resp = fs_client.get("/fs/scenario", params={"path": scenario_path})
        assert resp.status_code == 200
        assert resp.json()["conversations"] == []

    def test_traversal_attempt(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.get("/fs/scenario", params={"path": "../../etc/passwd"})
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_nonexistent_file(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.get("/fs/scenario", params={"path": "missing.json"})
        assert resp.status_code == 200
        assert "not found" in resp.json()["error"].lower()

    def test_invalid_json(self, fs_client: TestClient, fs_root: str) -> None:
        bad_path = os.path.join(fs_root, "bad.json")
        with open(bad_path, "w") as f:
            f.write("{invalid json")
        resp = fs_client.get("/fs/scenario", params={"path": bad_path})
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert "failed to load" in resp.json()["error"].lower()


class TestSaveConfig:
    def test_valid_save(self, fs_client: TestClient, fs_root: str) -> None:
        save_path = os.path.join(fs_root, "out_config.yaml")
        resp = fs_client.post(
            "/fs/config",
            json={"settings": {"key": "val"}, "path": save_path},
        )
        assert resp.status_code == 200
        assert resp.json()["path"] == save_path
        with open(save_path) as f:
            saved = yaml.safe_load(f)
        assert saved["key"] == "val"

    def test_path_outside_project(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.post(
            "/fs/config",
            json={"settings": {"k": "v"}, "path": "/tmp/evil.yaml"},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert "outside project" in resp.json()["error"].lower()

    def test_default_path_when_none(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.post(
            "/fs/config",
            json={"settings": {"x": "y"}},
        )
        assert resp.status_code == 200
        assert "path" in resp.json()
        assert resp.json()["path"].endswith("config_simulate.yaml")

    def test_filters_empty_values(self, fs_client: TestClient, fs_root: str) -> None:
        save_path = os.path.join(fs_root, "filtered.yaml")
        resp = fs_client.post(
            "/fs/config",
            json={
                "settings": {"keep": "yes", "drop_none": None, "drop_empty": ""},
                "path": save_path,
            },
        )
        assert resp.status_code == 200
        with open(save_path) as f:
            saved = yaml.safe_load(f)
        assert "keep" in saved
        assert "drop_none" not in saved
        assert "drop_empty" not in saved


class TestSaveScenario:
    def test_valid_save(self, fs_client: TestClient, fs_root: str) -> None:
        save_path = os.path.join(fs_root, "scenarios", "out.json")
        resp = fs_client.post(
            "/fs/scenario",
            json={"data": {"conversations": []}, "path": save_path},
        )
        assert resp.status_code == 200
        assert resp.json()["path"] == save_path
        with open(save_path) as f:
            saved = json.load(f)
        assert saved["conversations"] == []

    def test_path_outside_project(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.post(
            "/fs/scenario",
            json={"data": {}, "path": "/tmp/evil.json"},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert "outside project" in resp.json()["error"].lower()


class TestBrowseDirectory:
    def test_default_browse_returns_root(
        self, fs_client: TestClient, fs_root: str
    ) -> None:
        os.makedirs(os.path.join(fs_root, "subdir"))
        with open(os.path.join(fs_root, "file.txt"), "w") as f:
            f.write("hi")
        resp = fs_client.get("/fs/browse")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] == fs_root
        names = [e["name"] for e in body["entries"]]
        assert "subdir" in names
        assert "file.txt" in names

    def test_browse_subdirectory(self, fs_client: TestClient, fs_root: str) -> None:
        subdir = os.path.join(fs_root, "child")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "a.txt"), "w") as f:
            f.write("a")
        resp = fs_client.get("/fs/browse", params={"path": subdir})
        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] == subdir
        assert body["parent"] is not None

    def test_browse_outside_root_falls_back(
        self, fs_client: TestClient, fs_root: str
    ) -> None:
        resp = fs_client.get("/fs/browse", params={"path": "/tmp"})
        assert resp.status_code == 200
        assert resp.json()["current"] == fs_root


class TestGetProjectRoot:
    def test_returns_root(self, fs_client: TestClient, fs_root: str) -> None:
        resp = fs_client.get("/fs/root")
        assert resp.status_code == 200
        assert resp.json()["root"] == fs_root


class TestListConfigs:
    def test_discovers_yaml_files(self, fs_client: TestClient, fs_root: str) -> None:
        cfg_path = os.path.join(fs_root, "config_simulate.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump({"key": "val"}, f)
        resp = fs_client.get("/fs/configs")
        assert resp.status_code == 200
        configs = resp.json()["configs"]
        assert len(configs) >= 1
        paths = [c["path"] for c in configs]
        assert cfg_path in paths
