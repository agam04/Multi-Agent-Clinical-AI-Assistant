import json
from app.agents.core import MedicalAgentBase
from app.utils.model_registry import get_vision_model
from app.utils.prompts import coding_prompt
from app.graph.schema import WorkflowState
from app.utils.logger import get_logger
from app.utils.transforms import extract_json, blank_image
from langsmith.run_helpers import traceable
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate

logger = get_logger(__name__)


class DiagnosticCoderAgent(MedicalAgentBase):
    def __init__(self):
        super().__init__(name="DiagnosticCoderAgent")
        self.model, self.processor, self.config = get_vision_model()

    @traceable
    def infer(self, state: WorkflowState) -> str:
        clinical_note = state.payload.get("clinical_note", None)
        image = [state.payload.get("image") or blank_image()]
        logger.info(
            "DiagnosticCoderAgent.infer — has_note: %s, has_image: %s",
            clinical_note is not None,
            "image" in state.payload,
        )
        prompt = coding_prompt(clinical_note, image[0])
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        return generate(self.model, self.processor, formatted, image)

    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("Running %s", self.name)
        try:
            raw = self.infer(state).text
            logger.info("DiagnosticCoderAgent raw response received")
            codes = json.loads(extract_json(raw))
            logger.info("DiagnosticCoderAgent extracted %d codes", len(codes) if isinstance(codes, list) else 1)
            return WorkflowState(task="coding", payload=state.payload, result=codes, error=None)
        except Exception as e:
            logger.error("DiagnosticCoderAgent failed: %s", e)
            return WorkflowState(task="coding", payload=state.payload, result=None, error=str(e))
