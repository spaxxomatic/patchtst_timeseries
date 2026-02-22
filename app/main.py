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


@app.get("/sim-detail/{run_folder}", response_class=HTMLResponse)
async def sim_detail(request: Request, run_folder: str):
    folder = TRADESIMLOG / run_folder
    if not (folder / "sim_log.csv").exists():
        raise HTTPException(status_code=404, detail="sim_log.csv not found")
    return templates.TemplateResponse("sim_detail.html", {
        "request": request,
        "run_folder": run_folder,
    })


@app.get("/sim-data/{run_folder}")
async def sim_data_api(run_folder: str):
    import pandas as pd
    from ticker_data import get_ticker

    folder   = TRADESIMLOG / run_folder
    sim_path = folder / "sim_log.csv"
    par_path = folder / "tradesimparams.json"

    if not sim_path.exists():
        raise HTTPException(status_code=404, detail="sim_log.csv not found")

    params        = json.loads(par_path.read_text()) if par_path.exists() else {}
    symbol        = params.get("traded_symbol", "")
    trading_start = params.get("trading_start", "")
    trading_end   = params.get("trading_end", "")

    # ── Sim log ──────────────────────────────────────────────────────────
    df = pd.read_csv(sim_path, parse_dates=["date"])
    has_vix = "vix_gate" in df.columns

    sim_rows = []
    for _, row in df.iterrows():
        sim_rows.append({
            "date":            row["date"].strftime("%Y-%m-%d"),
            "pred_momentum":   _sf(row.get("pred_momentum")),
            "ci_lo_80":        _sf(row.get("ci_lo_80")),
            "ci_hi_80":        _sf(row.get("ci_hi_80")),
            "signal":          int(row["signal"]) if pd.notna(row.get("signal")) else 0,
            "in_market":       int(row["in_market"]) if pd.notna(row.get("in_market")) else 0,
            "portfolio_value": _sf(row.get("portfolio_value")),
            "position_return": _sf(row.get("position_return")) or 0.0,
            "vix_gate":        (str(row["vix_gate"]).lower() not in ("false", "0")) if has_vix else True,
        })

    # ── OHLCV price data ─────────────────────────────────────────────────
    ohlcv = []
    if symbol and trading_start and trading_end:
        try:
            price_df = get_ticker(symbol, trading_start, trading_end)
            for dt, pr in price_df.iterrows():
                ohlcv.append({
                    "date":  dt.strftime("%Y-%m-%d"),
                    "open":  round(float(pr["Open"]),  4),
                    "high":  round(float(pr["High"]),  4),
                    "low":   round(float(pr["Low"]),   4),
                    "close": round(float(pr["Close"]), 4),
                })
        except Exception:
            pass

    # ── Buy & Hold baseline ──────────────────────────────────────────────
    bh = []
    if ohlcv:
        c0 = ohlcv[0]["close"]
        for row in ohlcv:
            bh.append({"date": row["date"], "value": round(10_000.0 * row["close"] / c0, 4)})

    # ── Trade detection ──────────────────────────────────────────────────
    trades = _detect_trades(sim_rows)

    # ── Perf stats ───────────────────────────────────────────────────────
    perf = {}
    pf = folder / "perf_stats.json"
    if pf.exists():
        perf = json.loads(pf.read_text())

    return JSONResponse({
        "run_folder":  run_folder,
        "symbol":      symbol,
        "sim_log":     sim_rows,
        "ohlcv":       ohlcv,
        "bh":          bh,
        "trades":      trades,
        "perf_stats":  perf,
        "params":      {k: params[k] for k in [
            "THRESHOLD", "STOPLOSS_THRESHOLD", "FEE",
            "signal_horizon_steps", "trading_start", "trading_end",
        ] if k in params},
    })


def _sf(v):
    """Safe float: returns None for NaN/None/non-numeric."""
    try:
        f = float(v)
        return None if f != f else f   # NaN → None
    except Exception:
        return None


def _detect_trades(sim_rows: list[dict]) -> list[dict]:
    """Scan sim_log rows and return one dict per trade."""
    trades, current, prev_im = [], None, 0
    for i, row in enumerate(sim_rows):
        im = row["in_market"]
        if prev_im == 0 and im != 0:
            current = {
                "entry_date":      row["date"],
                "direction":       im,
                "entry_pred":      row["pred_momentum"],
                "entry_ci_lo":     row["ci_lo_80"],
                "entry_ci_hi":     row["ci_hi_80"],
                "entry_portfolio": row["portfolio_value"],
            }
        elif prev_im != 0 and im == 0 and current is not None:
            prev = sim_rows[i - 1]
            current.update({
                "exit_date":      row["date"],
                "final_return":   prev.get("position_return", 0.0),
                "exit_portfolio": prev.get("portfolio_value"),
                "is_stoploss":    prev.get("signal") == -2,
            })
            trades.append(current)
            current = None
        prev_im = im

    if current is not None and sim_rows:
        last = sim_rows[-1]
        current.update({
            "exit_date":      last["date"],
            "final_return":   last.get("position_return", 0.0),
            "exit_portfolio": last.get("portfolio_value"),
            "is_open":        True,
        })
        trades.append(current)
    return trades


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