"""
FastAPI dashboard — trade simulation results overview.

Run from project root:
    uvicorn app.main:app --reload --port 8000
"""

import json
import os
import shutil
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
from app.reports import load_reports, get_symbols, TRADESIMLOG, PROJECT_ROOT
from app.model_reports import load_model_reports, CHECKPOINTS
from lib.tradeparams import TradeSimParams
from simulator import _SIM_GENERATED_FILES, _SIM_LOCK_FILE

CHECKPOINTS_FAILURES = PROJECT_ROOT / "checkpoints_failures"
_SURFACE_LOCK      = "surface.lock"
_PRED_SURFACE_LOCK = "pred_surface.lock"

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
            # The signal fired on the previous row; in_market flips one row late
            # because the simulator captures day_current_pos before signal processing.
            sig = sim_rows[i - 1] if i > 0 else row
            current = {
                "entry_date":      sig["date"],
                "direction":       im,
                "entry_pred":      sig["pred_momentum"],
                "entry_ci_lo":     sig.get("ci_lo_80"),
                "entry_ci_hi":     sig.get("ci_hi_80"),
                "entry_portfolio": sig["portfolio_value"],
            }
        elif prev_im != 0 and im == 0 and current is not None:
            prev = sim_rows[i - 1]
            current.update({
                "exit_date":      prev["date"],   # same shift: exit signal was on prev row
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


# ─── Model delete ─────────────────────────────────────────────────────────────

@app.post("/delete-model/{model_folder:path}")
async def delete_model(model_folder: str):
    """Move a checkpoint folder to checkpoints_failures/."""
    src = CHECKPOINTS / model_folder
    if not src.exists():
        raise HTTPException(status_code=404, detail="Checkpoint folder not found")
    dest = CHECKPOINTS_FAILURES / model_folder
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return JSONResponse({"status": "moved", "dest": str(dest)})


# ─── Threshold surface study ──────────────────────────────────────────────────

def _find_tradesim_folder_for_model(model_folder: str) -> Path | None:
    """Return the most-recent tradesimlog folder that used this model."""
    matches = []
    for folder in TRADESIMLOG.iterdir():
        pf = folder / "tradesimparams.json"
        if not pf.exists():
            continue
        try:
            params = json.loads(pf.read_text())
            if model_folder in params.get("model_path", ""):
                matches.append(folder)
        except Exception:
            pass
    return max(matches, key=lambda p: p.name) if matches else None


@app.get("/surface/{model_folder:path}", response_class=HTMLResponse)
async def surface_page(request: Request, model_folder: str):
    return templates.TemplateResponse("surface.html", {
        "request":      request,
        "model_folder": model_folder,
    })


@app.get("/surface-status/{model_folder:path}")
async def surface_status(model_folder: str):
    study_dir    = CHECKPOINTS / model_folder / "surface_study"
    lock_file    = study_dir / _SURFACE_LOCK
    summary_file = study_dir / "surface_summary.json"

    if lock_file.exists():
        return JSONResponse({"status": "running"})
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        images  = {
            name: (study_dir / f"surface_{name}.png").exists()
            for name in ("3d", "slices", "hist")
        }
        return JSONResponse({"status": "done", "summary": summary, "images": images})
    return JSONResponse({"status": "pending"})


def _do_surface_run(model_folder: str) -> None:
    from thresh_surface import run_surface_study
    from lib.tradeparams import TradeSimParams

    os.chdir(PROJECT_ROOT)
    study_dir = CHECKPOINTS / model_folder / "surface_study"
    lock_file = study_dir / _SURFACE_LOCK
    try:
        sim_folder = _find_tradesim_folder_for_model(model_folder)
        if sim_folder is None:
            raise RuntimeError(f"No tradesimlog entry found for model {model_folder}")

        params = TradeSimParams.load_from_folder(sim_folder)
        if not hasattr(params, "model_storage_folder") or params.model_storage_folder is None:
            params.model_storage_folder = Path(params.model_path) / "model"

        run_surface_study(params, n_trials=600, output_dir=study_dir)
    except Exception as exc:
        # Persist error message so the UI can surface it
        (study_dir / "surface_error.txt").write_text(str(exc))
    finally:
        lock_file.unlink(missing_ok=True)


@app.post("/surface-run/{model_folder:path}")
async def surface_run(model_folder: str, background_tasks: BackgroundTasks):
    study_dir = CHECKPOINTS / model_folder / "surface_study"
    lock_file = study_dir / _SURFACE_LOCK
    if lock_file.exists():
        return JSONResponse({"status": "already_running"})
    if (study_dir / "surface_summary.json").exists():
        return JSONResponse({"status": "already_done"})
    sim_folder = _find_tradesim_folder_for_model(model_folder)
    if sim_folder is None:
        raise HTTPException(status_code=422, detail="No tradesimlog run found for this model — run a simulation first")
    study_dir.mkdir(parents=True, exist_ok=True)
    lock_file.touch()
    background_tasks.add_task(_do_surface_run, model_folder)
    return JSONResponse({"status": "started"})


@app.get("/surface-image/{model_folder:path}")
async def surface_image(model_folder: str, name: str = Query(...)):
    """Serve a surface study PNG. name ∈ {3d, slices, hist}."""
    img = CHECKPOINTS / model_folder / "surface_study" / f"surface_{name}.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img, media_type="image/png")


# ─── Prediction accuracy surface study ───────────────────────────────────────

@app.get("/pred-surface/{model_folder:path}", response_class=HTMLResponse)
async def pred_surface_page(request: Request, model_folder: str):
    return templates.TemplateResponse("pred_surface.html", {
        "request":      request,
        "model_folder": model_folder,
    })


@app.get("/pred-surface-status/{model_folder:path}")
async def pred_surface_status(model_folder: str):
    study_dir    = CHECKPOINTS / model_folder / "pred_surface"
    lock_file    = study_dir / _PRED_SURFACE_LOCK
    summary_file = study_dir / "pred_surface_summary.json"
    error_file   = study_dir / "pred_surface_error.txt"

    if lock_file.exists():
        return JSONResponse({"status": "running"})
    if error_file.exists():
        return JSONResponse({"status": "error", "message": error_file.read_text()})
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        images  = {
            name: (study_dir / f"pred_surface_{name}.png").exists()
            for name in ("heatmaps", "lines", "hist", "coverage")
        }
        return JSONResponse({"status": "done", "summary": summary, "images": images})
    return JSONResponse({"status": "pending"})


def _do_pred_surface_run(model_folder: str) -> None:
    from optuna_pred_surface import run_pred_surface_study

    os.chdir(PROJECT_ROOT)
    study_dir = CHECKPOINTS / model_folder / "pred_surface"
    lock_file = study_dir / _PRED_SURFACE_LOCK
    try:
        sim_folder = _find_tradesim_folder_for_model(model_folder)
        if sim_folder is None:
            raise RuntimeError(f"No tradesimlog entry found for model {model_folder}")

        params = TradeSimParams.load_from_folder(sim_folder)
        if not hasattr(params, "model_storage_folder") or params.model_storage_folder is None:
            object.__setattr__(params, "model_storage_folder", Path(params.model_path) / "model")

        run_pred_surface_study(params, n_trials=1000, output_dir=study_dir)
    except Exception as exc:
        study_dir.mkdir(parents=True, exist_ok=True)
        (study_dir / "pred_surface_error.txt").write_text(str(exc))
    finally:
        lock_file.unlink(missing_ok=True)


@app.post("/pred-surface-run/{model_folder:path}")
async def pred_surface_run(model_folder: str, background_tasks: BackgroundTasks):
    study_dir = CHECKPOINTS / model_folder / "pred_surface"
    lock_file = study_dir / _PRED_SURFACE_LOCK
    if lock_file.exists():
        return JSONResponse({"status": "already_running"})
    if (study_dir / "pred_surface_summary.json").exists():
        return JSONResponse({"status": "already_done"})
    sim_folder = _find_tradesim_folder_for_model(model_folder)
    if sim_folder is None:
        raise HTTPException(
            status_code=422,
            detail="No tradesimlog run found for this model — run a simulation first",
        )
    study_dir.mkdir(parents=True, exist_ok=True)
    lock_file.touch()
    background_tasks.add_task(_do_pred_surface_run, model_folder)
    return JSONResponse({"status": "started"})


@app.get("/pred-surface-image/{model_folder:path}")
async def pred_surface_image(model_folder: str, name: str = Query(...)):
    """Serve a pred-surface PNG. name ∈ {heatmaps, lines, hist, coverage}."""
    img = CHECKPOINTS / model_folder / "pred_surface" / f"pred_surface_{name}.png"
    if not img.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img, media_type="image/png")


def _do_pred_surface_replot(model_folder: str) -> None:
    from optuna_pred_surface import replot_from_existing
    os.chdir(PROJECT_ROOT)
    output_dir = CHECKPOINTS / model_folder / "pred_surface"
    lock_file  = output_dir / _PRED_SURFACE_LOCK
    try:
        replot_from_existing(output_dir)
    except Exception as exc:
        (output_dir / "pred_surface_error.txt").write_text(str(exc))
    finally:
        lock_file.unlink(missing_ok=True)


@app.post("/pred-surface-replot/{model_folder:path}")
async def pred_surface_replot(model_folder: str, background_tasks: BackgroundTasks):
    """Regenerate plots from an existing CSV — no inference or Optuna."""
    study_dir = CHECKPOINTS / model_folder / "pred_surface"
    lock_file = study_dir / _PRED_SURFACE_LOCK
    if not (study_dir / "pred_surface_results.csv").exists():
        raise HTTPException(status_code=422, detail="No results CSV — run the full study first")
    if lock_file.exists():
        return JSONResponse({"status": "already_running"})
    lock_file.touch()
    background_tasks.add_task(_do_pred_surface_replot, model_folder)
    return JSONResponse({"status": "started"})


# ─── Regime Change Monitor ────────────────────────────────────────────────────

@app.get("/regime", response_class=HTMLResponse)
async def regime_page(request: Request):
    from app.regime_monitor import load_regime_data, sector_summary, SECTOR_ORDER, SECTOR_COLORS
    from collections import defaultdict

    rows      = load_regime_data()
    summaries = {s["sector"]: s for s in sector_summary(rows)}

    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[row["sector"]].append(row)

    groups = []
    for sector in SECTOR_ORDER:
        if sector in grouped:
            groups.append({
                "sector":  sector,
                "color":   SECTOR_COLORS.get(sector, "#495057"),
                "rows":    grouped[sector],
                "summary": summaries.get(sector, {}),
            })

    n_inverted   = sum(1 for r in rows if r["status"] in ("inverted", "inverted_broad"))
    n_aligned    = sum(1 for r in rows if r["status"] in ("aligned", "aligned_broad"))
    as_of_dates  = [r["as_of"] for r in rows if r["as_of"] != "—"]
    max_as_of    = max(as_of_dates) if as_of_dates else "—"

    return templates.TemplateResponse("regime_monitor.html", {
        "request":    request,
        "groups":     groups,
        "total":      len(rows),
        "n_inverted": n_inverted,
        "n_aligned":  n_aligned,
        "max_as_of":  max_as_of,
    })


@app.get("/regime-data")
async def regime_data_api():
    from app.regime_monitor import load_regime_data, sector_summary
    rows      = load_regime_data()
    summaries = sector_summary(rows)
    return JSONResponse({"rows": rows, "summaries": summaries})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)