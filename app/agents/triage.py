from PIL import Image
import numpy as np
from app.utils.logger import get_logger
from app.graph.schema import WorkflowState
from app.utils.prompts import triage_prompt
from langsmith.run_helpers import traceable
from app.agents.core import MedicalAgentBase
from app.utils.model_registry import get_vision_model
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm import generate

logger = get_logger(__name__)


class TriageAgent(MedicalAgentBase):
    def __init__(self):
        super().__init__(name="TriageAgent")
        self.model, self.processor, self.config = get_vision_model()

    @traceable
    def infer(self, state: WorkflowState) -> str:
        image = [
            state.payload["image"]
            if "image" in state.payload
            else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        ]
        note = state.payload.get("note", None)
        logger.info(
            "TriageAgent.infer — has_image: %s, has_note: %s",
            "image" in state.payload,
            note is not None,
        )
        prompt = triage_prompt(note, image[0])
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        return generate(self.model, self.processor, formatted, image)

    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("Running %s", self.name)
        decision = self.infer(state).text.lower().strip()
        logger.info("TriageAgent decision: %s", decision)

        if decision == "coding":
            new_payload = {**state.payload, "clinical_note": state.payload.get("note", "")}
            return WorkflowState(task="coding", payload=new_payload, result=None, error=None)
        elif decision == "documentation":
            new_payload = {**state.payload, "transcript": state.payload.get("note", "")}
            return WorkflowState(task="documentation", payload=new_payload, result=None, error=None)
        elif decision == "imaging":
            new_payload = {**state.payload, "clinical_note": state.payload.get("note", "")}
            return WorkflowState(task="imaging", payload=new_payload, result=None, error=None)
        else:
            logger.error("TriageAgent returned unrecognised decision: %s", decision)
            return WorkflowState(
                task=None,
                payload=state.payload,
                result=None,
                error=f"Triage could not classify input: '{decision}'",
            )
