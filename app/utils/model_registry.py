import threading
from mlx_vlm import load
from mlx_vlm.utils import load_config
from app.config.settings import app_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_processor = None
_config = None
_lock = threading.Lock()


def get_vision_model():
    global _model, _processor, _config

    if _model is not None and _processor is not None:
        return _model, _processor, _config

    with _lock:
        if _model is None or _processor is None:
            logger.info("Loading %s...", app_config.MODEL_ID)
            _model, _processor = load(app_config.MODEL_ID)
            _config = load_config(app_config.MODEL_ID)

    return _model, _processor, _config
