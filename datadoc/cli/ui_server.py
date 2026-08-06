import os
import json
from pathlib import Path
from dataclasses import dataclass
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from datadoc.core.engine import DATADOC
from datadoc.core.pipeline import DataDocError, DataDocPipeline, PipelineConfig
from datadoc.core.agent import AgenticEngineer

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
    doc: DATADOC
    agent: Optional[AgenticEngineer]
    pipeline: Optional[DataDocPipeline] = None
    profile: Optional[dict] = None
    plan: Optional[dict] = None


_sessions: dict[str, SessionState] = {}


class PluginRequest(BaseModel):
    plugins: List[str]


class ChatRequest(BaseModel):
    message: str


class PipelineRequest(BaseModel):
    target: Optional[str] = None
    task: str = "auto"
    drop_identifiers: bool = False
    scaling: str = "auto"
    clip_outliers: bool = False


def init_server(file_path: str):
    """Initializes the local dashboard session before server startup."""
    from datadoc.cli.app import load_dataset

    doc = load_dataset(file_path)
    model = os.getenv("DATADOC_MODEL", "groq/llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY", "")
    _sessions["local"] = SessionState(
        doc=doc,
        agent=AgenticEngineer(metadata=doc._extract_metadata(), api_key=api_key, model=model),
    )


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
    doc = _state(session_id).doc
    return {
        "file": doc.file_path,
        "rows": doc.df.height,
        "columns": doc.df.width,
        "metadata": doc._extract_metadata(),
    }


@app.get("/api/dataset/recommend")
def recommend_pipeline(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    doc = _state(session_id).doc
    doc.revert()
    return {"plugins": doc.list_plugins()}


@app.get("/api/pipeline/profile")
def pipeline_profile(
    target: Optional[str] = None,
    session_id: str = Header("local", alias="X-DATADOC-SESSION"),
):
    state = _state(session_id)
    try:
        state.profile = (
            DataDocPipeline(PipelineConfig(target=target)).profile(state.doc._original_df).to_dict()
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
        state.plan = DataDocPipeline(_pipeline_config(req)).plan(state.doc._original_df).to_dict()
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
        state.pipeline = DataDocPipeline(_pipeline_config(req)).fit(state.doc._original_df)
        state.profile = state.pipeline.profile_.to_dict() if state.pipeline.profile_ else None
        state.plan = state.pipeline.plan_.to_dict() if state.pipeline.plan_ else None
        transformed = state.pipeline.transform(state.doc._original_df)
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
        transformed = state.pipeline.transform(state.doc._original_df)
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
    from fastapi.responses import Response

    return Response(
        content=code,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=pipeline.py"},
    )


@app.get("/api/dataset/export/csv")
def export_csv(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    doc = _state(session_id).doc

    from fastapi.responses import Response

    state = _state(session_id)
    output = state.pipeline.transform(doc._original_df) if state.pipeline else doc.df
    csv_bytes = output.write_csv()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_data.csv"},
    )


@app.get("/api/dataset/export/code")
def export_code(session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    doc = _state(session_id).doc

    from fastapi.responses import Response

    code = doc.pipeline()
    return Response(
        content=code,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=pipeline.py"},
    )


@app.post("/api/dataset/plugins")
def apply_plugins(req: PluginRequest, session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    state = _state(session_id)
    doc = state.doc
    available = {plugin.name: plugin for plugin in doc.plugins}
    if len(req.plugins) != len(set(req.plugins)) or any(
        name not in available for name in req.plugins
    ):
        raise HTTPException(
            status_code=422, detail="Plugin list contains unknown or duplicate plugins."
        )
    positions = {name: index for index, name in enumerate(req.plugins)}
    for name in req.plugins:
        for dependency in available[name].dependencies:
            if dependency in positions and positions[dependency] > positions[name]:
                raise HTTPException(status_code=422, detail=f"{name} must run after {dependency}.")
    doc.revert()

    results = []
    for plugin_name in req.plugins:
        try:
            res = doc.apply_plugin_by_name(plugin_name)
            results.append({"plugin": plugin_name, "status": "success", "detail": res})
        except Exception as e:
            results.append({"plugin": plugin_name, "status": "error", "detail": str(e)})
            break  # Stop applying subsequent plugins if one fails

    # Send back missing values breakdown, distributions, etc.
    clean_df = doc.df
    diff = doc.compare(clean_df)

    return {"results": results, "shape": (clean_df.height, clean_df.width), "diff": diff}


@app.post("/api/agent/chat")
def agent_chat(req: ChatRequest, session_id: str = Header("local", alias="X-DATADOC-SESSION")):
    agent = _state(session_id).agent
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized.")
    try:
        reply = agent.chat_step(req.message)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
