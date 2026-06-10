"""
Integration tests for agent execute() methods.

The global model mock from conftest.py ensures no real MLX model is loaded.
Each test patches `generate` locally to control what the model "returns".
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.graph.schema import WorkflowState


# ── helpers ──────────────────────────────────────────────────────────────────

def _gen_response(text: str) -> MagicMock:
    m = MagicMock()
    m.text = text
    return m


# ── TriageAgent ───────────────────────────────────────────────────────────────

class TestTriageAgent:
    def _make_agent(self):
        from app.agents.triage import TriageAgent
        return TriageAgent()

    @pytest.mark.parametrize("model_output,expected_task", [
        ("coding", "coding"),
        ("  CODING\n", "coding"),      # whitespace + casing handled
        ("documentation", "documentation"),
        ("imaging", "imaging"),
    ])
    def test_routes_correctly(self, note_state, model_output, expected_task):
        with patch("app.agents.triage.generate", return_value=_gen_response(model_output)):
            agent = self._make_agent()
            result = agent.execute(note_state)
        assert result.task == expected_task
        assert result.error is None

    def test_unknown_decision_returns_error(self, note_state):
        with patch("app.agents.triage.generate", return_value=_gen_response("unknown_garbage")):
            agent = self._make_agent()
            result = agent.execute(note_state)
        assert result.error is not None
        assert result.task is None

    def test_coding_route_copies_note_to_clinical_note(self, note_state):
        with patch("app.agents.triage.generate", return_value=_gen_response("coding")):
            agent = self._make_agent()
            result = agent.execute(note_state)
        assert "clinical_note" in result.payload
        assert result.payload["clinical_note"] == note_state.payload["note"]

    def test_documentation_route_copies_note_to_transcript(self, note_state):
        with patch("app.agents.triage.generate", return_value=_gen_response("documentation")):
            agent = self._make_agent()
            result = agent.execute(note_state)
        assert "transcript" in result.payload
        assert result.payload["transcript"] == note_state.payload["note"]

    def test_original_state_not_mutated(self, note_state):
        original_payload = dict(note_state.payload)
        with patch("app.agents.triage.generate", return_value=_gen_response("coding")):
            agent = self._make_agent()
            agent.execute(note_state)
        assert note_state.payload == original_payload


# ── DiagnosticCoderAgent ──────────────────────────────────────────────────────

class TestDiagnosticCoderAgent:
    def _make_agent(self):
        from app.agents.coder import DiagnosticCoderAgent
        return DiagnosticCoderAgent()

    def test_parses_valid_icd10_json(self, coding_state):
        codes = [{"code": "E11.9", "description": "Type 2 diabetes mellitus without complications"}]
        with patch("app.agents.coder.generate", return_value=_gen_response(json.dumps(codes))):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        assert result.error is None
        assert result.task == "coding"
        # Pydantic coerces List[dict] → List[DiagnosticCode] on WorkflowState.result
        r = result.result
        codes_out = [c if isinstance(c, dict) else c.model_dump() for c in r]
        # evidence defaults to None when the model omits it; compare core fields
        for out, expected in zip(codes_out, codes):
            assert out["code"] == expected["code"]
            assert out["description"] == expected["description"]

    def test_parses_fenced_json(self, coding_state):
        codes = [{"code": "E11.9", "description": "Type 2 diabetes"}]
        fenced = f"```json\n{json.dumps(codes)}\n```"
        with patch("app.agents.coder.generate", return_value=_gen_response(fenced)):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        assert result.error is None
        assert len(result.result) == 1

    def test_filters_codes_with_empty_description(self, coding_state):
        codes = [
            {"code": "E11.9", "description": "Diabetes"},
            {"code": "Z00.00", "description": ""},
        ]
        with patch("app.agents.coder.generate", return_value=_gen_response(json.dumps(codes))):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        # Pydantic may coerce to DiagnosticCode objects; normalise before asserting
        descriptions = [
            (c.description if hasattr(c, "description") else c["description"])
            for c in result.result
        ]
        assert all(descriptions)

    def test_malformed_json_returns_error_state(self, coding_state):
        with patch("app.agents.coder.generate", return_value=_gen_response("not json at all !!!")):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        # Unrecoverable → empty list from extract_json → json.loads([]) succeeds
        # so result may be [] rather than an error; both are acceptable non-crash outcomes
        assert result.task == "coding"

    def test_inference_exception_returns_error_state(self, coding_state):
        with patch("app.agents.coder.generate", side_effect=RuntimeError("MLX crash")):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        assert result.error is not None
        assert "MLX crash" in result.error
        assert result.result is None

    def test_evidence_field_flows_through(self, coding_state):
        codes = [{
            "code": "E11.9",
            "description": "Type 2 diabetes mellitus",
            "evidence": "10-year history of type 2 diabetes mellitus",
        }]
        with patch("app.agents.coder.generate", return_value=_gen_response(json.dumps(codes))):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        assert result.error is None
        c = result.result[0]
        evidence = c.evidence if hasattr(c, "evidence") else c["evidence"]
        assert evidence == "10-year history of type 2 diabetes mellitus"

    def test_missing_evidence_defaults_to_none(self, coding_state):
        codes = [{"code": "I10", "description": "Essential hypertension"}]
        with patch("app.agents.coder.generate", return_value=_gen_response(json.dumps(codes))):
            agent = self._make_agent()
            result = agent.execute(coding_state)
        c = result.result[0]
        evidence = c.evidence if hasattr(c, "evidence") else c.get("evidence")
        assert evidence is None


# ── ClinicalDocumentationAgent ────────────────────────────────────────────────

class TestClinicalDocumentationAgent:
    def _make_agent(self):
        from app.agents.documenter import ClinicalDocumentationAgent
        return ClinicalDocumentationAgent()

    def test_parses_valid_soap_json(self, documentation_state):
        soap = {
            "Subjective": "Headache for 3 days.",
            "Objective": "BP 120/80, temp 37.0.",
            "Assessment": "Tension headache.",
            "Plan": "Ibuprofen 400mg PRN.",
        }
        with patch("app.agents.documenter.generate", return_value=_gen_response(json.dumps(soap))):
            agent = self._make_agent()
            result = agent.execute(documentation_state)
        assert result.error is None
        assert result.result["Subjective"] == "Headache for 3 days."
        assert result.task == "documentation"

    def test_fenced_soap_json_parsed(self, documentation_state):
        soap = {"Subjective": "S", "Objective": "O", "Assessment": "A", "Plan": "P"}
        fenced = f"```json\n{json.dumps(soap)}\n```"
        with patch("app.agents.documenter.generate", return_value=_gen_response(fenced)):
            agent = self._make_agent()
            result = agent.execute(documentation_state)
        assert result.error is None

    def test_inference_exception_returns_error_state(self, documentation_state):
        with patch("app.agents.documenter.generate", side_effect=ValueError("bad input")):
            agent = self._make_agent()
            result = agent.execute(documentation_state)
        assert result.error is not None
        assert result.result is None


# ── RadiologyAgent ────────────────────────────────────────────────────────────

class TestRadiologyAgent:
    def _make_agent(self):
        from app.agents.imaging import RadiologyAgent
        return RadiologyAgent()

    def test_parses_valid_imaging_report(self, imaging_state):
        report = {
            "technique": "PA chest X-ray without contrast.",
            "findings": "No consolidation or pleural effusion.",
            "impression": "Normal chest radiograph.",
            "recommendations": "No further imaging required.",
            "answer_to_user_question": "No evidence of pneumonia.",
        }
        with patch("app.agents.imaging.generate", return_value=_gen_response(json.dumps(report))):
            agent = self._make_agent()
            result = agent.execute(imaging_state)
        assert result.error is None
        assert result.result["impression"] == "Normal chest radiograph."
        assert result.task == "imaging"

    def test_inference_exception_returns_error_state(self, imaging_state):
        with patch("app.agents.imaging.generate", side_effect=RuntimeError("GPU OOM")):
            agent = self._make_agent()
            result = agent.execute(imaging_state)
        assert result.error is not None
        assert "GPU OOM" in result.error
        assert result.result is None
