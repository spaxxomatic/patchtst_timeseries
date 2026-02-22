"""
FastAPI dashboard — trade simulation results overview.

Run from project root:
    uvicorn app.main:app --reload --port 8000
"""

import json
import os
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
from app.reports import load_reports, get_symbols, TRADESIMLOG, PROJECT_ROOT
from app.model_reports import load_model_reports, CHECKPOINTS
from simulator import _SIM_GENERATED_FILES, _SIM_LOCK_FILE

app = FastAPI(title="TradeSimulator Dashboard")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    symbol: str = Query(default=None),
):
    all_rows = load_reports()
    symbols  = get_symbols(all_rows)

    # Default to first symbol if none selected
    if symbol is None and symbols:
        symbol = symbols[0]

    rows = [r for r in all_rows if r["symbol"] == symbol] if symbol else all_rows

    return templates.TemplateResponse("index.html", {
        "request": request,
        "rows":    rows,
        "symbols": symbols,
        "selected_symbol": symbol,
    })


    
@app.get("/image/{run_folder}")
async def run_image(run_folder: str):
    img = TRADESIMLOG / run_folder / "result.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img, media_type="image/png")


@app.get("/cv-image/{run_folder}")
async def cv_image(run_folder: str):
    """Serve cv_backtest.png from the model_path recorded in that run's params."""
    params_file = TRADESIMLOG / run_folder / "tradesimparams.json"
    if not params_file.exists():
        raise HTTPException(status_code=404, detail="Run params not found")
    model_path = json.loads(params_file.read_text()).get("model_path")
    if not model_path:
        raise HTTPException(status_code=404, detail="model_path not set for this run")
    img = PROJECT_ROOT / model_path / "cv_backtest.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="cv_backtest.png not found")
    return FileResponse(img, media_type="image/png")


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    rows = load_model_reports()
    return templates.TemplateResponse("models.html", {
        "request": request,
        "rows":    rows,
    })


@app.get("/model-cv-image/{model_folder:path}")
async def model_cv_image(model_folder: str):
    img = CHECKPOINTS / model_folder / "cv_backtest.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="cv_backtest.png not found")
    return FileResponse(img, media_type="image/png")


@app.get("/model-train-image/{model_folder:path}")
async def model_train_image(model_folder: str):
    img = CHECKPOINTS / model_folder / "training_curves.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="training_curves.png not found")
    return FileResponse(img, media_type="image/png")


@app.post("/delete-sim/{run_folder}")
async def delete_sim(run_folder: str):
    folder = TRADESIMLOG / run_folder
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    for fname in _SIM_GENERATED_FILES + [_SIM_LOCK_FILE]:
        (folder / fname).unlink(missing_ok=True)
    return JSONResponse({"status": "deleted"})


@app.get("/sim-status/{run_folder}")
async def sim_status(run_folder: str):
    folder = TRADESIMLOG / run_folder
    if (folder / _SIM_LOCK_FILE).exists():
        return JSONResponse({"status": "running"})
    if (folder / "sim_log.csv").exists():
        return JSONResponse({"status": "done"})
    return JSONResponse({"status": "pending"})


def _do_rerun(run_folder: str):
    from lib.tradeparams import TradeSimParams
    from simulator import run_simulation
    os.chdir(PROJECT_ROOT)
    folder = TRADESIMLOG / run_folder
    for fname in _SIM_GENERATED_FILES + [_SIM_LOCK_FILE]:
        (folder / fname).unlink(missing_ok=True)
    params = TradeSimParams.load_from_folder(folder)
    object.__setattr__(params, 'model_storage_folder', Path(params.model_path) / "model")
    run_simulation(params)


@app.post("/rerun-sim/{run_folder}")
async def rerun_sim(run_folder: str, background_tasks: BackgroundTasks):
    folder = TRADESIMLOG / run_folder
    if not (folder / "tradesimparams.json").exists():
        raise HTTPException(status_code=404, detail="Run params not found")
    if (folder / _SIM_LOCK_FILE).exists():
        raise HTTPException(status_code=409, detail="Simulation already running")
    background_tasks.add_task(_do_rerun, run_folder)
    return JSONResponse({"status": "started"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)