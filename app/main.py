"""
FastAPI dashboard — trade simulation results overview.

Run from project root:
    uvicorn app.main:app --reload --port 8000
"""

import json
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
from app.reports import load_reports, get_symbols, TRADESIMLOG, PROJECT_ROOT

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)