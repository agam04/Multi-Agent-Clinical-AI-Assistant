import json
from app.agents.core import MedicalAgentBase
from app.utils.model_registry import get_vision_model
from app.utils.prompts import imaging_prompt
from app.graph.schema import WorkflowState
from app.utils.logger import get_logger
from app.utils.transforms import extract_json, blank_image
from langsmith.run_helpers import traceable
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate

logger = get_logger(__name__)


class RadiologyAgent(MedicalAgentBase):
    def __init__(self):
        super().__init__(name="RadiologyAgent")
        self.model, self.processor, self.config = get_vision_model()

    @traceable
    def infer(self, state: WorkflowState) -> str:
        raw_image = state.payload.get("image", None)
        note = state.payload.get("note", None)
        image = [raw_image or blank_image()]
        logger.info("RadiologyAgent.infer — has_image: %s, has_note: %s", raw_image is not None, note is not None)
        prompt = imaging_prompt(raw_image, note)
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        return generate(self.model, self.processor, formatted, image)

    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("Running %s", self.name)
        try:
            raw = self.infer(state).text
            logger.info("RadiologyAgent raw response received")
            report = json.loads(extract_json(raw))
            return WorkflowState(task="imaging", payload=state.payload, result=report, error=None)
        except Exception as e:
            logger.error("RadiologyAgent failed: %s", e)
            return WorkflowState(task="imaging", payload=state.payload, result=None, error=str(e))
