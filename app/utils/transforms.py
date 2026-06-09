from PIL import Image
import io
import re
import json
from app.utils.logger import get_logger

logger = get_logger(__name__)


def decode_image_upload(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Uploaded file could not be decoded as an image.") from e


def extract_json(raw: str) -> str:
    cleaned = re.sub(r"^```json\s*|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            data = [entry for entry in data if entry.get("description")]
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = [entry for entry in value if entry.get("description")]
        logger.info("extract_json: parsed successfully")
        return json.dumps(data)
    except Exception as e:
        logger.error("JSON extraction failed: %s", e)
        matches = re.findall(
            r'\{[^{}]*"code"\s*:\s*"[^"]*",\s*"description"\s*:\s*"[^"]*"\s*\}',
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
        logger.info("extract_json: recovered %d partial objects", len(recovered))
        return json.dumps(recovered)
