from PIL import Image
from typing import Optional
from app.utils.logger import get_logger
from app.utils.transforms import blank_image as _blank

logger = get_logger(__name__)


def triage_prompt(note: Optional[str], image: Optional[Image.Image]) -> str:
    img = image if image is not None else _blank()
    return f"""You are a clinical intake classifier. Examine the inputs below and decide
which downstream pipeline should handle them.

Available pipelines:
  coding        — accepts clinical notes describing a patient encounter; extracts diagnosis codes
  documentation — accepts a dialogue or transcript between clinician and patient; generates a structured note
  imaging       — accepts a medical image (X-ray, MRI, CT, etc.) that requires radiological interpretation

Decision rules:
  • If the text contains dialogue turns (e.g. "Doctor:", "Patient:", "Clinician:", "Nurse:") → documentation
  • If the text is a clinical summary, discharge note, or list of diagnoses/conditions with no dialogue → coding
  • If a medical image is the primary input → imaging

Reply with exactly one word — no punctuation, no explanation:
  coding | documentation | imaging

Input:
  text: {note if note else "none"}
  image: {img}
"""


def coding_prompt(clinical_note: str, image: Optional[Image.Image]) -> str:
    img = image if image is not None else _blank()
    return f"""You are a certified clinical coder specialising in ICD-10-CM.
Your job is to read the clinical note below and return the relevant diagnostic codes.

Guidelines:
  - Include codes from the A00–R99 range (diseases, symptoms, conditions)
  - Only include Z-codes (Z00–Z99) when they carry direct clinical relevance
  - Draw codes from both the primary diagnosis and any documented comorbidities
  - Never duplicate a code; each should appear once
  - If a condition is ambiguous, omit rather than guess
  - For each code, copy the exact span of text from the note (word-for-word, no
    paraphrasing) into "evidence" so a reviewer can verify the assignment
  - Output ONLY a valid JSON array — no markdown, no prose, no extra keys

Required format (double-quoted keys and values, codes derived solely from the note below):
[
  {{"code": "<ICD-10 code>", "description": "<condition as documented>", "evidence": "<verbatim phrase from the note supporting this code>"}},
  {{"code": "<ICD-10 code>", "description": "<condition as documented>", "evidence": "<verbatim phrase from the note supporting this code>"}}
]

Clinical note:
{clinical_note}
Image context: {img}
"""


def imaging_prompt(image: Optional[Image.Image], question: Optional[str] = None) -> str:
    img = image if image is not None else _blank()
    return f"""You are a board-certified radiologist reviewing a medical image.
Produce a structured radiology report based solely on what you observe.

Return ONLY a JSON object with these exact keys (double-quoted, no extra fields):

{{
  "technique":               "Imaging modality, views obtained, and whether contrast was used.",
  "findings":                "Systematic description of all observable structures and any abnormalities.",
  "impression":              "Concise clinical summary and differential diagnoses.",
  "recommendations":         "Suggested next steps — further imaging, clinical correlation, or follow-up.",
  "answer_to_user_question": "Direct answer to the question below, or null if no question was given."
}}

No markdown. No prose outside the JSON object.

Image: {img}
Question: {question if question else "none"}
"""


def documentation_prompt(transcript: str, image: Optional[Image.Image]) -> str:
    img = image if image is not None else _blank()
    return f"""You are a clinical documentation specialist. Convert the medical encounter
transcript below into a SOAP note.

Section definitions:
  Subjective   — What the patient reports: chief complaint, symptom history, duration, severity,
                 relevant past medical and social history. Use bullet points.
  Objective    — Clinician-observed data: vital signs, physical exam findings, lab values,
                 imaging results. Use bullet points.
  Assessment   — The clinician's diagnostic interpretation: working and differential diagnoses.
  Plan         — Agreed next steps: medications, tests ordered, referrals, patient education,
                 follow-up schedule. Use bullet points.

Rules:
  - Only include information present in the transcript; do not fabricate details
  - Keep language concise and clinical
  - Return ONLY a valid JSON object with the four keys below (double-quoted)

{{
  "Subjective":  "...",
  "Objective":   "...",
  "Assessment":  "...",
  "Plan":        "..."
}}

Transcript:
{transcript}
Image context: {img}
"""
