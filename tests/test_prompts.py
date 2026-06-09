"""Unit tests for app/utils/prompts.py"""
import pytest
from app.utils.prompts import (
    triage_prompt,
    coding_prompt,
    imaging_prompt,
    documentation_prompt,
)


class TestTriagePrompt:
    def test_returns_string(self):
        result = triage_prompt("fever and cough", None)
        assert isinstance(result, str)

    def test_note_appears_in_prompt(self):
        note = "Unique clinical note xyz123"
        result = triage_prompt(note, None)
        assert note in result

    def test_none_note_handled(self):
        result = triage_prompt(None, None)
        assert isinstance(result, str)

    def test_all_three_pipeline_names_mentioned(self):
        result = triage_prompt("some note", None)
        assert "coding" in result
        assert "documentation" in result
        assert "imaging" in result


class TestCodingPrompt:
    def test_returns_string(self):
        result = coding_prompt("Patient has diabetes", None)
        assert isinstance(result, str)

    def test_clinical_note_appears(self):
        note = "Acute uncomplicated appendicitis confirmed."
        result = coding_prompt(note, None)
        assert note in result

    def test_json_format_instructed(self):
        result = coding_prompt("some note", None)
        assert "JSON" in result or "json" in result

    def test_none_note_handled(self):
        result = coding_prompt(None, None)
        assert isinstance(result, str)


class TestImagingPrompt:
    def test_returns_string(self):
        result = imaging_prompt(None, None)
        assert isinstance(result, str)

    def test_question_appears_when_provided(self):
        q = "Is there a pleural effusion?"
        result = imaging_prompt(None, q)
        assert q in result

    def test_no_question_handled(self):
        result = imaging_prompt(None, None)
        assert isinstance(result, str)

    def test_expected_output_fields_mentioned(self):
        result = imaging_prompt(None, None)
        for field in ("technique", "findings", "impression", "recommendations"):
            assert field in result


class TestDocumentationPrompt:
    def test_returns_string(self):
        result = documentation_prompt("Doctor: Hello. Patient: Hi.", None)
        assert isinstance(result, str)

    def test_transcript_appears(self):
        transcript = "Doctor: Any chest pain? Patient: Yes, since morning."
        result = documentation_prompt(transcript, None)
        assert transcript in result

    def test_soap_sections_mentioned(self):
        result = documentation_prompt("some transcript", None)
        for section in ("Subjective", "Objective", "Assessment", "Plan"):
            assert section in result

    def test_json_output_instructed(self):
        result = documentation_prompt("transcript", None)
        assert "JSON" in result or "json" in result
