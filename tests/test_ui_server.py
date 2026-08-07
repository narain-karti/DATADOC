"""Smoke tests for the optional local dashboard API."""

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from datadoc.cli.ui_server import app, init_server


@pytest.fixture(autouse=True)
def _init_test_server(tmp_path):
    csv_path = tmp_path / "test.csv"
    pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]}).write_csv(csv_path)
    init_server(str(csv_path))


def test_pipeline_api_round_trip() -> None:
    client = TestClient(app)
    headers = {"X-DATADOC-SESSION": "local"}
    request = {"target": None, "task": "auto", "scaling": "none"}

    profile = client.get("/api/pipeline/profile", headers=headers)
    assert profile.status_code == 200
    assert "roles" in profile.json()

    plan = client.post("/api/pipeline/plan", json=request, headers=headers)
    assert plan.status_code == 200
    assert "operations" in plan.json()

    fitted = client.post("/api/pipeline/fit", json=request, headers=headers)
    assert fitted.status_code == 200
    assert fitted.json()["fitted"] is True

    preview = client.get("/api/pipeline/preview", headers=headers)
    assert preview.status_code == 200
    assert "rows" in preview.json()

    exported = client.get("/api/pipeline/export/code", headers=headers)
    assert exported.status_code == 200
    assert "DataDocPipeline" in exported.text


def test_pipeline_api_rejects_invalid_session() -> None:
    client = TestClient(app)
    response = client.get("/api/pipeline/profile", headers={"X-DATADOC-SESSION": "does-not-exist"})
    assert response.status_code == 404
