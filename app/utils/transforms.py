from PIL import Image
import io
import re
import json
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BLANK_IMAGE = None


def blank_image() -> Image.Image:
    """Reusable 224×224 black placeholder for text-only requests.

    MedGemma's chat template always expects one image; when a request carries
    only text we feed this dummy so the vision path stays well-formed. Cached
    so every agent shares a single instance instead of re-allocating.
    """
    global _BLANK_IMAGE
    if _BLANK_IMAGE is None:
        _BLANK_IMAGE = Image.new("RGB", (224, 224), (0, 0, 0))
    return _BLANK_IMAGE


def decode_image_upload(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Uploaded file could not be decoded as an image.") from e


def _first_json_object(text: str):
    """Return the first complete, balanced top-level JSON object in *text*.

    Scans character by character tracking brace depth while respecting string
    literals (so braces inside values don't throw off the count). Used to
    recover SOAP / imaging responses that the model wrapped in prose or a
    partial code fence. Returns None if no balanced object parses cleanly.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
    return None


def extract_json(raw: str) -> str:
    cleaned = re.sub(r"^```json\s*|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            data = [entry for entry in data if isinstance(entry, dict) and entry.get("description")]
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = [entry for entry in value if isinstance(entry, dict) and entry.get("description")]
        logger.info("extract_json: parsed successfully")
        return json.dumps(data)
    except Exception as e:
        logger.error("JSON extraction failed: %s", e)

        # 1) Recover ICD-10 code objects from a truncated / noisy array response.
        #    Matches any single-level object carrying both "code" and "description",
        #    tolerating extra keys (e.g. "evidence") in any position.
        matches = re.findall(
            r'\{[^{}]*?"code"\s*:\s*"[^"]*"[^{}]*?"description"\s*:\s*"[^"]*"[^{}]*?\}',
            cleaned,
        )
        recovered = []
        for m in matches:
            try:
                obj = json.loads(m)
                if obj.get("description"):
                    recovered.append(obj)
            except Exception:
                continue
        if recovered:
            logger.info("extract_json: recovered %d code objects", len(recovered))
            return json.dumps(recovered)

        # 2) Fall back to the first balanced JSON object (SOAP / imaging shapes).
        obj = _first_json_object(cleaned)
        if obj is not None:
            logger.info("extract_json: recovered a JSON object via brace matching")
            return json.dumps(obj)

        logger.info("extract_json: nothing recoverable")
        return json.dumps([])
