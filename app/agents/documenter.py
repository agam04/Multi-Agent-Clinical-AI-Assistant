import json
from app.agents.core import MedicalAgentBase
from app.utils.model_registry import get_vision_model
from app.utils.prompts import documentation_prompt
from app.graph.schema import WorkflowState
from app.utils.logger import get_logger
from app.utils.transforms import extract_json, blank_image
from langsmith.run_helpers import traceable
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate

logger = get_logger(__name__)


class ClinicalDocumentationAgent(MedicalAgentBase):
    def __init__(self):
        super().__init__(name="ClinicalDocumentationAgent")
        self.model, self.processor, self.config = get_vision_model()

    @traceable
    def infer(self, state: WorkflowState) -> str:
        transcript = state.payload.get("transcript", "")
        image = [state.payload.get("image") or blank_image()]
        logger.info("ClinicalDocumentationAgent.infer — transcript length: %d chars", len(transcript))
        prompt = documentation_prompt(transcript, image[0])
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        return generate(self.model, self.processor, formatted, image)

    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("Running %s", self.name)
        try:
            raw = self.infer(state).text
            logger.info("ClinicalDocumentationAgent raw response received")
            note = json.loads(extract_json(raw))
            return WorkflowState(task="documentation", payload=state.payload, result=note, error=None)
        except Exception as e:
            logger.error("ClinicalDocumentationAgent failed: %s", e)
            return WorkflowState(task="documentation", payload=state.payload, result=None, error=str(e))
