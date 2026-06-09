from typing import TypedDict, Literal, Union, Optional, List
from PIL import Image as PILImage
from pydantic import BaseModel
from app.api.schemas import DiagnosticCode


class ICD10Input(TypedDict):
    clinical_note: str


class TranscriptInput(TypedDict):
    transcript: str


class ImagingInput(TypedDict):
    image: PILImage.Image
    clinical_note: Optional[str]


class WorkflowState(BaseModel):
    task: Optional[Literal["coding", "documentation", "imaging"]]
    payload: dict
    result: Optional[Union[str, List[DiagnosticCode], dict]]
    error: Optional[str]
