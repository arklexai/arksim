# SPDX-License-Identifier: Apache-2.0
"""Tests for results API endpoints via FastAPI TestClient."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arksim.ui.api.routes_results import router


def _make_app(root: str) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    # Set up app.state.arksim with evaluate.result = None by default
    evaluate_state = SimpleNamespace(result=None, output_dir=None)
    app.state.arksim = SimpleNamespace(evaluate=evaluate_state)
    return app


@pytest.fixture()
def results_root(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    root = str(tmp_path)
    monkeypatch.setattr("arksim.ui.api.routes_filesystem.PROJECT_ROOT", root)
    monkeypatch.setattr("arksim.ui.api.routes_results.PROJECT_ROOT", root)
    return root


@pytest.fixture()
def results_client(results_root: str) -> TestClient:
    app = _make_app(results_root)
    return TestClient(app)


class TestGetResults:
    def test_with_valid_dir_and_results_file(
        self, results_client: TestClient, results_root: str
    ) -> None:
        results_dir = os.path.join(results_root, "output", "run1")
        os.makedirs(results_dir)
        results_file = os.path.join(results_dir, "evaluation_results.json")
        data = {"scores": [0.9]}
        with open(results_file, "w") as f:
            json.dump(data, f)
        resp = results_client.get("/results", params={"dir": results_dir})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"]["scores"] == [0.9]
        assert body["output_dir"] == results_dir

    def test_traversal_attempt(
        self, results_client: TestClient, results_root: str
    ) -> None:
        resp = results_client.get("/results", params={"dir": "/tmp/evil"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] is None

    def test_without_dir_param(
        self, results_client: TestClient, results_root: str
    ) -> None:
        resp = results_client.get("/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] is None
        assert body["output_dir"] is None

    def test_dir_without_results_file(
        self, results_client: TestClient, results_root: str
    ) -> None:
        empty_dir = os.path.join(results_root, "empty")
        os.makedirs(empty_dir)
        resp = results_client.get("/results", params={"dir": empty_dir})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] is None

    def test_in_memory_results(self, results_root: str) -> None:
        app = _make_app(results_root)
        mock_result = SimpleNamespace(
            model_dump=lambda: {"metric": 1.0},
        )
        app.state.arksim.evaluate.result = mock_result
        app.state.arksim.evaluate.output_dir = "/some/dir"
        client = TestClient(app)
        resp = client.get("/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"]["metric"] == 1.0
        assert body["output_dir"] == "/some/dir"


class TestGetReport:
    def test_valid_report(self, results_client: TestClient, results_root: str) -> None:
        report_dir = os.path.join(results_root, "output")
        os.makedirs(report_dir)
        report_path = os.path.join(report_dir, "final_report.html")
        with open(report_path, "w") as f:
            f.write("<html>report</html>")
        resp = results_client.get("/results/report", params={"dir": report_dir})
        assert resp.status_code == 200
        assert "report" in resp.text

    def test_missing_report(
        self, results_client: TestClient, results_root: str
    ) -> None:
        empty_dir = os.path.join(results_root, "no_report")
        os.makedirs(empty_dir)
        resp = results_client.get("/results/report", params={"dir": empty_dir})
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"].lower()

    def test_traversal_attempt(
        self, results_client: TestClient, results_root: str
    ) -> None:
        resp = results_client.get("/results/report", params={"dir": "/tmp/evil"})
        assert resp.status_code == 403


class TestGetResultFile:
    def test_valid_file(self, results_client: TestClient, results_root: str) -> None:
        file_dir = os.path.join(results_root, "output")
        os.makedirs(file_dir)
        file_path = os.path.join(file_dir, "final_report.html")
        with open(file_path, "w") as f:
            f.write("<html>content</html>")
        resp = results_client.get(
            "/results/file", params={"dir": file_dir, "name": "final_report.html"}
        )
        assert resp.status_code == 200

    def test_disallowed_filename(
        self, results_client: TestClient, results_root: str
    ) -> None:
        resp = results_client.get(
            "/results/file",
            params={"dir": results_root, "name": "secret.txt"},
        )
        assert resp.status_code == 403
        assert "not allowed" in resp.json()["error"].lower()

    def test_traversal_attempt(
        self, results_client: TestClient, results_root: str
    ) -> None:
        resp = results_client.get(
            "/results/file",
            params={"dir": "/tmp/evil", "name": "final_report.html"},
        )
        assert resp.status_code == 403

    def test_missing_file(self, results_client: TestClient, results_root: str) -> None:
        empty_dir = os.path.join(results_root, "empty")
        os.makedirs(empty_dir)
        resp = results_client.get(
            "/results/file",
            params={"dir": empty_dir, "name": "final_report.html"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"].lower()
