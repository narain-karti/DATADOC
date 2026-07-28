import os
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from datadoc.core.engine import DATADOC
from datadoc.core.agent import AgenticEngineer

app = FastAPI(title="DATADOC UI Server")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset instance (simplified for single-user local dashboard)
_doc_instance: Optional[DATADOC] = None
_agent_instance: Optional[AgenticEngineer] = None

class PluginRequest(BaseModel):
    plugins: List[str]

class ChatRequest(BaseModel):
    message: str

def init_server(file_path: str):
    """Initializes the global DATADOC instance before server startup."""
    global _doc_instance, _agent_instance
    from datadoc.cli.app import load_dataset
    _doc_instance = load_dataset(file_path)
    # Get model and api_key from env or defaults
    model = os.getenv("DATADOC_MODEL", "groq/llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY", "")
    _agent_instance = AgenticEngineer(
        metadata=_doc_instance._extract_metadata(),
        api_key=api_key,
        model=model
    )

@app.get("/api/dataset/metadata")
def get_metadata():
    if not _doc_instance:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")
    return {
        "file": _doc_instance.file_path,
        "rows": _doc_instance.df.height,
        "columns": _doc_instance.df.width,
        "metadata": _doc_instance._extract_metadata()
    }

@app.get("/api/dataset/recommend")
def recommend_pipeline():
    if not _doc_instance:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")
    _doc_instance.revert() # Ensure we're recommending from raw state
    return {"plugins": _doc_instance.list_plugins()}

@app.get("/api/dataset/export/csv")
def export_csv():
    if not _doc_instance:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")
    
    from fastapi.responses import Response
    csv_bytes = _doc_instance.df.write_csv()
    return Response(content=csv_bytes, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=cleaned_data.csv"
    })

@app.get("/api/dataset/export/code")
def export_code():
    if not _doc_instance:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")
    
    from fastapi.responses import Response
    code = _doc_instance.pipeline()
    return Response(content=code, media_type="text/plain", headers={
        "Content-Disposition": "attachment; filename=pipeline.py"
    })

@app.post("/api/dataset/plugins")
def apply_plugins(req: PluginRequest):
    if not _doc_instance:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")
    
    # We revert to raw, then apply exactly the ordered list of plugins
    _doc_instance.revert()
    
    results = []
    for plugin_name in req.plugins:
        try:
            res = _doc_instance.apply_plugin_by_name(plugin_name)
            results.append({"plugin": plugin_name, "status": "success", "detail": res})
        except Exception as e:
            results.append({"plugin": plugin_name, "status": "error", "detail": str(e)})
            break # Stop applying subsequent plugins if one fails
            
    # Send back missing values breakdown, distributions, etc.
    clean_df = _doc_instance.df
    diff = _doc_instance.compare(clean_df) 
    
    return {
        "results": results,
        "shape": (clean_df.height, clean_df.width),
        "diff": diff
    }

@app.post("/api/agent/chat")
def agent_chat(req: ChatRequest):
    if not _agent_instance:
        raise HTTPException(status_code=400, detail="Agent not initialized.")
    try:
        reply = _agent_instance.chat_step(req.message)
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
        return HTMLResponse("<h1>DATADOC UI Server is running!</h1><p>React build not found in web/dist.</p>")
