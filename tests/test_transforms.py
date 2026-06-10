"""Unit tests for app/utils/transforms.py"""
import json
import io
import pytest
import numpy as np
from PIL import Image

from app.utils.transforms import extract_json, decode_image_upload


# ── extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_clean_array_passes_through(self):
        raw = '[{"code": "K35.80", "description": "Acute appendicitis"}]'
        result = json.loads(extract_json(raw))
        assert result == [{"code": "K35.80", "description": "Acute appendicitis"}]

    def test_strips_markdown_json_fence(self):
        raw = '```json\n[{"code": "A01", "description": "Typhoid"}]\n```'
        result = json.loads(extract_json(raw))
        assert result[0]["code"] == "A01"

    def test_strips_plain_backtick_fence(self):
        raw = '```\n[{"code": "B01", "description": "Varicella"}]\n```'
        result = json.loads(extract_json(raw))
        assert result[0]["code"] == "B01"

    def test_filters_entries_with_empty_description(self):
        raw = json.dumps([
            {"code": "K35.80", "description": "Appendicitis"},
            {"code": "Z00.00", "description": ""},
            {"code": "R10.9", "description": "Abdominal pain"},
        ])
        result = json.loads(extract_json(raw))
        codes = [r["code"] for r in result]
        assert "Z00.00" not in codes
        assert "K35.80" in codes
        assert "R10.9" in codes

    def test_filters_entries_with_missing_description(self):
        raw = json.dumps([
            {"code": "K35.80", "description": "Appendicitis"},
            {"code": "Z99.99"},
        ])
        result = json.loads(extract_json(raw))
        assert len(result) == 1
        assert result[0]["code"] == "K35.80"

    def test_preserves_apostrophes_in_description(self):
        raw = json.dumps([{"code": "M79.3", "description": "Patient's panniculitis"}])
        result = json.loads(extract_json(raw))
        assert "Patient's panniculitis" in result[0]["description"]

    def test_dict_with_nested_list_filters_empty_descriptions(self):
        raw = json.dumps({
            "codes": [
                {"code": "A", "description": "Valid"},
                {"code": "B", "description": ""},
            ]
        })
        result = json.loads(extract_json(raw))
        assert len(result["codes"]) == 1
        assert result["codes"][0]["code"] == "A"

    def test_partial_recovery_from_truncated_json(self):
        # Model truncated output — only partial JSON remains
        raw = '[{"code": "K35.80", "description": "Appendicitis"}, {"code": "R10'
        result = json.loads(extract_json(raw))
        assert isinstance(result, list)
        assert any(r["code"] == "K35.80" for r in result)

    def test_partial_recovery_with_evidence_field(self):
        # Truncated array where each object also carries an "evidence" key
        raw = ('[{"code": "E11.9", "description": "Diabetes", '
               '"evidence": "type 2 diabetes mellitus"}, {"code": "R10')
        result = json.loads(extract_json(raw))
        assert any(r["code"] == "E11.9" for r in result)
        recovered = next(r for r in result if r["code"] == "E11.9")
        assert recovered["evidence"] == "type 2 diabetes mellitus"

    def test_recovers_soap_object_wrapped_in_prose(self):
        # Whole-string parse fails (prose around it); brace matcher recovers the object
        raw = ('Here is the note you requested:\n'
               '{"Subjective": "Headache", "Objective": "Normal", '
               '"Assessment": "Tension", "Plan": "Rest"} — hope this helps!')
        result = json.loads(extract_json(raw))
        assert result["Subjective"] == "Headache"
        assert result["Plan"] == "Rest"

    def test_unrecoverable_garbage_returns_empty_list(self):
        raw = "Sorry, I cannot provide medical codes."
        result = json.loads(extract_json(raw))
        assert result == []

    def test_empty_string_returns_empty_list(self):
        result = json.loads(extract_json(""))
        assert result == []


# ── decode_image_upload ──────────────────────────────────────────────────────

class TestDecodeImageUpload:
    def _png_bytes(self, size=(32, 32)) -> bytes:
        img = Image.fromarray(np.zeros((*size, 3), dtype="uint8"))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_png_returns_pil_image(self):
        result = decode_image_upload(self._png_bytes())
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(self):
        result = decode_image_upload(self._png_bytes())
        assert result.mode == "RGB"

    def test_invalid_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="decoded as an image"):
            decode_image_upload(b"not an image")

    def test_empty_bytes_raises_value_error(self):
        with pytest.raises(ValueError):
            decode_image_upload(b"")
