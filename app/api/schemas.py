from pydantic import BaseModel
from typing import List, Optional, Literal, Union, TypeAlias


class DiagnosticCode(BaseModel):
    code: str
    description: str


class CodingResponse(BaseModel):
    agent: Literal["coding"]
    result: List[DiagnosticCode]


class ClinicalNote(BaseModel):
    Subjective: str
    Objective: str
    Assessment: str
    Plan: str


class DocumentationResponse(BaseModel):
    agent: Literal["documentation"]
    result: ClinicalNote


class ImagingReport(BaseModel):
    technique: str
    findings: str
    impression: str
    recommendations: str
    answer_to_user_question: Optional[str] = None


class ImagingResponse(BaseModel):
    agent: Literal["imaging"]
    result: ImagingReport


class ApiError(BaseModel):
    error: str


ApiResponse: TypeAlias = Union[
    CodingResponse,
    DocumentationResponse,
    ImagingResponse,
    ApiError,
]
