import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langsmith import Client

from app.api.analyze import router as analyze_router

load_dotenv()

app = FastAPI(title="Multi-Agent Medical System")

_langsmith_key = os.getenv("LANGCHAIN_API_KEY")
if _langsmith_key:
    langsmith_client = Client()

app.include_router(analyze_router, prefix="/api")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join("app/static", "index.html"))
