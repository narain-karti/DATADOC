import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import polars as pl

from datadoc.core.pipeline import (
    DataDocError,
    DataDocPipeline,
    PipelineConfig,
    profile_dataset,
    read_dataset,
)

app = FastAPI(title="DATADOC UI Server")

LOCAL_ORIGINS = os.getenv(
    "DATADOC_UI_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class SessionState:
    df: pl.DataFrame
    file_path: str
    pipeline: Optional[DataDocPipeline] = None
    profile: Optional[dict] = None
    plan: Optional[dict] = None


_sessions: dict[str, SessionState] = {}




class PipelineRequest(BaseModel):
    target: Optional[str] = None
    task: str = "auto"
    drop_identifiers: bool = False
    scaling: str = "auto"
    clip_outliers: bool = False


def init_server(file_path: str):
    """Initializes the local dashboard session before server startup."""
    df = read_dataset(file_path)
    _sessions["local"] = SessionState(df=df, file_path=file_path)


def _state(session_id: str) -> SessionState:
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Dataset session not found.")
    return state


def _pipeline_config(req: PipelineRequest) -> PipelineConfig:
    if req.task not in {"auto", "classification", "regression"}:
        raise HTTPException(
            status_code=422, detail="task must be auto, classification, or regression."
        )
    if req.scaling not in {"auto", "none", "standard", "robust"}:
        raise HTTPException(
            status_code=422, detail="scaling must be auto, none, standard, or robust."
        )
    return PipelineConfig(
        target=req.target,
        task=req.task,
        drop_identifiers=req.drop_identifiers,
        scaling=req.scaling,
        clip_outliers=req.clip_outliers,
    )


@app.get("/api/dataset/metadata")
def get_metadata(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    state = _state(session_id)
    df = state.df
    profile = profile_dataset(df, PipelineConfig())
    return {
        "file": state.file_path,
        "rows": df.height,
        "columns": df.width,
        "metadata": profile.to_dict(),
    }


@app.get("/api/pipeline/profile")
def pipeline_profile(
    target: Optional[str] = None,
    session_id: str = Header("local", alias="X-DATADOC-SESSION"),
):
    state = _state(session_id)
    try:
        state.profile = (
            DataDocPipeline(PipelineConfig(target=target)).profile(state.df).to_dict()
        )
        return state.profile
    except DataDocError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/pipeline/plan")
def pipeline_plan(
    req: PipelineRequest,
    session_id: str = Header("local", alias="X-DATADOC-SESSION"),
):
    state = _state(session_id)
    try:
        state.plan = DataDocPipeline(_pipeline_config(req)).plan(state.df).to_dict()
        return state.plan
    except DataDocError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/pipeline/fit")
def pipeline_fit(
    req: PipelineRequest,
    session_id: str = Header("local", alias="X-DATADOC-SESSION"),
):
    state = _state(session_id)
    try:
        state.pipeline = DataDocPipeline(_pipeline_config(req)).fit(state.df)
        state.profile = state.pipeline.profile_.to_dict() if state.pipeline.profile_ else None
        state.plan = state.pipeline.plan_.to_dict() if state.pipeline.plan_ else None
        transformed = state.pipeline.transform(state.df)
        return {
            "profile": state.profile,
            "plan": state.plan,
            "input_schema": state.pipeline.input_schema_,
            "output_schema": state.pipeline.output_schema_,
            "rows": transformed.height,
            "columns": transformed.width,
            "fitted": True,
        }
    except DataDocError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/pipeline/preview")
def pipeline_preview(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    state = _state(session_id)
    if not state.pipeline:
        raise HTTPException(status_code=400, detail="Fit a pipeline before requesting a preview.")
    try:
        transformed = state.pipeline.transform(state.df)
        return {"schema": state.pipeline.output_schema_, "rows": transformed.head(8).to_dicts()}
    except DataDocError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/pipeline/export/code")
def pipeline_export_code(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    state = _state(session_id)
    if not state.pipeline:
        raise HTTPException(status_code=400, detail="Fit a pipeline before exporting code.")
    artifact = json.dumps(state.pipeline.to_dict(), indent=2)
    code = f"""import json
import polars as pl
from datadoc.core.pipeline import DataDocPipeline, PipelineConfig, read_dataset

ARTIFACT = json.loads({artifact!r})

def transform_file(input_path: str, output_path: str) -> None:
    pipeline = DataDocPipeline(PipelineConfig(**ARTIFACT["config"]))
    pipeline.input_schema_ = ARTIFACT["input_schema"]
    pipeline.output_schema_ = ARTIFACT["output_schema"]
    pipeline.state_ = ARTIFACT["state"]
    pipeline.fitted_ = True
    transformed = pipeline.transform(read_dataset(input_path))
    if output_path.endswith(".parquet"):
        transformed.write_parquet(output_path)
    else:
        transformed.write_csv(output_path)
"""
    return Response(
        content=code,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=pipeline.py"},
    )


@app.get("/api/dataset/export/csv")
def export_csv(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    state = _state(session_id)
    output = state.pipeline.transform(state.df) if state.pipeline else state.df
    csv_bytes = output.write_csv()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"},
    )


# Mount React App (if exists)
dist_dir = Path(__file__).parent.parent.parent / "web" / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="web")
else:

    @app.get("/")
    def read_root():
        return HTMLResponse(
            "<h1>DATADOC UI Server is running!</h1><p>React build not found in web/dist.</p>"
        )
