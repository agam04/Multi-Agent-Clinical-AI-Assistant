import asyncio
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.graph.pipeline import assemble_pipeline
from app.graph.schema import WorkflowState
from app.utils.transforms import decode_image_upload
from app.api.schemas import (
    CodingResponse,
    DocumentationResponse,
    ImagingResponse,
    ApiError,
    DiagnosticCode,
    ClinicalNote,
    ImagingReport,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
pipeline = assemble_pipeline()


@router.post("/analyze")
async def analyze(note: str = Form(None), image: UploadFile = File(None)):
    if not note and not image:
        return JSONResponse(status_code=400, content={"error": "No input provided."})

    try:
        logger.info(
            "Received request — note: %s, image: %s",
            "yes" if note else "no",
            image.filename if image else "none",
        )
        state = WorkflowState(task=None, payload={}, result=None, error=None)

        if note:
            state.payload["note"] = note

        if image and image.filename:
            contents = await image.read()
            state.payload["image"] = decode_image_upload(contents)

        raw_output = await asyncio.to_thread(pipeline.invoke, state)

        if isinstance(raw_output, dict):
            output = WorkflowState(**raw_output)
        elif isinstance(raw_output, WorkflowState):
            output = raw_output
        else:
            return JSONResponse(status_code=500, content={"error": "Unexpected output type from pipeline."})

        if output.error:
            return JSONResponse(status_code=500, content={"error": output.error})

        if output.task == "coding":
            # WorkflowState may have already coerced dicts to DiagnosticCode via Pydantic
            codes = [
                c if isinstance(c, DiagnosticCode) else DiagnosticCode.model_validate(c)
                for c in output.result
            ]
            return CodingResponse(agent="coding", result=codes)

        elif output.task == "documentation":
            note_obj = ClinicalNote(
                Subjective=output.result.get("Subjective", ""),
                Objective=output.result.get("Objective", ""),
                Assessment=output.result.get("Assessment", ""),
                Plan=output.result.get("Plan", ""),
            )
            return DocumentationResponse(agent="documentation", result=note_obj)

        elif output.task == "imaging":
            report = ImagingReport(
                technique=output.result.get("technique", ""),
                findings=output.result.get("findings", ""),
                impression=output.result.get("impression", ""),
                recommendations=output.result.get("recommendations", ""),
                answer_to_user_question=output.result.get("answer_to_user_question", None),
            )
            return ImagingResponse(agent="imaging", result=report)

        else:
            return JSONResponse(status_code=500, content={"error": "Pipeline returned unknown task type."})

    except Exception as e:
        logger.error("Unhandled error in /analyze: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
