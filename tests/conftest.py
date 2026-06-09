"""
Shared fixtures and global mocks.

Two layers of mocking run before any app code is imported:

1. MLX system mocks — mlx_vlm and related C extensions don't exist in the test
   environment (they're Apple-Silicon-only, not in CI deps). We stub them in
   sys.modules so that `from mlx_vlm import generate` resolves to a MagicMock
   instead of raising ImportError.

2. Model-registry mock — patches get_vision_model() so agents never try to
   load a real model. Applied at module level so it is active before any test
   file imports the application modules that trigger model loading.
"""
import sys
from unittest.mock import MagicMock, patch
import pytest

# ── 1. Stub MLX / mlx_vlm platform packages ─────────────────────────────────
for _mod in (
    "mlx", "mlx.core", "mlx.nn",
    "mlx_lm",
    "mlx_vlm", "mlx_vlm.utils", "mlx_vlm.prompt_utils",
    "mlx_metal",
):
    sys.modules.setdefault(_mod, MagicMock())

# ── 2. Stub model registry ───────────────────────────────────────────────────
_fake_model = MagicMock(name="mlx_model")
_fake_processor = MagicMock(name="mlx_processor")
_fake_config = MagicMock(name="mlx_config")

patch(
    "app.utils.model_registry.get_vision_model",
    return_value=(_fake_model, _fake_processor, _fake_config),
).start()

# Safe default for generate / apply_chat_template in every agent module.
# Import each module first so the attributes exist, then patch them.
_DEFAULT_RESPONSE = MagicMock()
_DEFAULT_RESPONSE.text = "[]"

import app.agents.triage
import app.agents.coder
import app.agents.documenter
import app.agents.imaging

for _agent_module in ("triage", "coder", "documenter", "imaging"):
    patch(f"app.agents.{_agent_module}.generate", return_value=_DEFAULT_RESPONSE).start()
    patch(f"app.agents.{_agent_module}.apply_chat_template", return_value="<prompt>").start()


# ── Reusable fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def blank_workflow_state():
    from app.graph.schema import WorkflowState
    return WorkflowState(task=None, payload={}, result=None, error=None)


@pytest.fixture()
def note_state():
    from app.graph.schema import WorkflowState
    return WorkflowState(
        task=None,
        payload={"note": "Patient presents with fever and cough for 3 days."},
        result=None,
        error=None,
    )


@pytest.fixture()
def coding_state():
    from app.graph.schema import WorkflowState
    return WorkflowState(
        task="coding",
        payload={
            "note": "Diagnosis: Type 2 diabetes mellitus.",
            "clinical_note": "Diagnosis: Type 2 diabetes mellitus.",
        },
        result=None,
        error=None,
    )


@pytest.fixture()
def documentation_state():
    from app.graph.schema import WorkflowState
    return WorkflowState(
        task="documentation",
        payload={
            "note": "Doctor: How long have you had the headache? Patient: Three days.",
            "transcript": "Doctor: How long have you had the headache? Patient: Three days.",
        },
        result=None,
        error=None,
    )


@pytest.fixture()
def imaging_state():
    from PIL import Image
    import numpy as np
    from app.graph.schema import WorkflowState
    img = Image.fromarray(np.zeros((64, 64, 3), dtype="uint8"))
    return WorkflowState(
        task="imaging",
        payload={"image": img, "note": "Evaluate for pneumonia."},
        result=None,
        error=None,
    )
