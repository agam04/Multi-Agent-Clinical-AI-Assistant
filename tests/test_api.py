"""
FastAPI endpoint tests for POST /api/analyze.

The pipeline is mocked so no real model inference runs.
"""
import json
import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _png_bytes() -> bytes:
    img = Image.fromarray(np.zeros((32, 32, 3), dtype="uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_state(task, result=None, error=None):
    from app.graph.schema import WorkflowState
    return WorkflowState(task=task, payload={}, result=result, error=error)


@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_no_input_returns_400(self, client):
        response = client.post("/api/analyze")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_empty_note_no_image_returns_400(self, client):
        response = client.post("/api/analyze", data={"note": ""})
        assert response.status_code == 400


# ── Successful routing ────────────────────────────────────────────────────────

class TestSuccessfulRouting:
    def test_coding_response_shape(self, client):
        codes = [{"code": "E11.9", "description": "Type 2 diabetes"}]
        output = _make_state("coding", result=codes)

        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = output
            response = client.post("/api/analyze", data={"note": "Diabetes note"})

        assert response.status_code == 200
        body = response.json()
        assert body["agent"] == "coding"
        assert body["result"][0]["code"] == "E11.9"

    def test_documentation_response_shape(self, client):
        soap = {
            "Subjective": "Headache.",
            "Objective": "Normal exam.",
            "Assessment": "Tension headache.",
            "Plan": "Rest and analgesia.",
        }
        output = _make_state("documentation", result=soap)

        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = output
            response = client.post("/api/analyze", data={"note": "transcript"})

        assert response.status_code == 200
        body = response.json()
        assert body["agent"] == "documentation"
        assert body["result"]["Subjective"] == "Headache."

    def test_imaging_response_shape(self, client):
        report = {
            "technique": "PA CXR",
            "findings": "Clear lungs.",
            "impression": "Normal.",
            "recommendations": "None.",
            "answer_to_user_question": None,
        }
        output = _make_state("imaging", result=report)

        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = output
            response = client.post(
                "/api/analyze",
                data={"note": "evaluate lungs"},
                files={"image": ("xray.png", _png_bytes(), "image/png")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["agent"] == "imaging"
        assert body["result"]["impression"] == "Normal."

    def test_image_only_request_accepted(self, client):
        report = {
            "technique": "MRI", "findings": "f",
            "impression": "i", "recommendations": "r",
            "answer_to_user_question": None,
        }
        output = _make_state("imaging", result=report)

        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = output
            response = client.post(
                "/api/analyze",
                files={"image": ("mri.png", _png_bytes(), "image/png")},
            )

        assert response.status_code == 200


# ── Error paths ───────────────────────────────────────────────────────────────

class TestErrorPaths:
    def test_pipeline_error_returns_500(self, client):
        output = _make_state(None, error="Model returned unclassifiable output")

        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = output
            response = client.post("/api/analyze", data={"note": "some note"})

        assert response.status_code == 500
        assert "error" in response.json()

    def test_pipeline_exception_returns_500(self, client):
        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.side_effect = RuntimeError("Unexpected crash")
            response = client.post("/api/analyze", data={"note": "some note"})

        assert response.status_code == 500
        assert "error" in response.json()

    def test_invalid_image_bytes_returns_500(self, client):
        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = _make_state("imaging", result={
                "technique": "", "findings": "", "impression": "",
                "recommendations": "", "answer_to_user_question": None
            })
            response = client.post(
                "/api/analyze",
                files={"image": ("bad.png", b"not an image", "image/png")},
            )

        assert response.status_code == 500

    def test_null_task_falls_through_to_500(self, client):
        # task=None is valid on WorkflowState (Optional) but hits the final else branch
        output = _make_state(None, result=None)

        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = output
            response = client.post("/api/analyze", data={"note": "note"})

        assert response.status_code == 500

    def test_non_state_pipeline_output_returns_500(self, client):
        # Pipeline returns something that is neither dict nor WorkflowState
        with patch("app.api.analyze.pipeline") as mock_pipeline:
            mock_pipeline.invoke.return_value = 42
            response = client.post("/api/analyze", data={"note": "note"})

        assert response.status_code == 500
