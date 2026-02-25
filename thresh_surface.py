"""thresh_surface.py
─────────────────────────────────────────────────────────────────────────────
Explore the trading yield surface over
  (THRESHOLD, STOPLOSS_THRESHOLD, TRAILING_STOP_THRESHOLD)
without re-running the neural network for every trial.

Workflow
────────
1. Build a predictions cache once  (model inference, one call per test day).
   Saved to  <model_path>/predictions_cache.pkl
2. Run an Optuna study with a quasi-random (QMC) sampler for good space
   coverage — not optimisation.
3. Save results to <model_path>/surface_study/surface_results.csv and
   render diagnostic plots in the same folder.

A smooth surface  → robust trading edge (parameters don't matter much).
A jagged surface  → fragile / overfit edge (cherry-picked thresholds).

Usage
─────
    # Full run — build cache if missing, then explore
    python thresh_surface.py --symbol MSFT

    # Only build / refresh the cache (e.g. after retraining)
    python thresh_surface.py --symbol MSFT --cache-only

    # Reload an existing cache and re-run the study with more trials
    python thresh_surface.py --symbol MSFT --n-trials 500

    # Use an existing tradesimlog folder as the source of params
    python thresh_surface.py --from-folder tradesimlog/MSFT_20260101_120000
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt

from lib.tradeparams import TradeSimParams, TradeSimulData, Trader
from neuralforecast import NeuralForecast

# ─── Constants (must match simulator.py) ──────────────────────────────────────
SIGNAL_TRIGGER_STOPLOSS = -2
SIGNAL_TRIGGER_TP       =  2
_FALLBACK_INPUT_LEN     = 130
CACHE_FILENAME          = "predictions_cache.pkl"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_input_size(model_path: str) -> int:
    summary_file = Path(model_path) / "optuna_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        input_size = summary.get("best_params", {}).get("input_size")
        if input_size is not None:
            return int(input_size)
    return _FALLBACK_INPUT_LEN


def _silence_lightning() -> None:
    for name in ("pytorch_lightning", "neuralforecast", "lightning.pytorch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"pytorch_lightning\..*")
    warnings.filterwarnings("ignore", module=r"lightning\..*")


# ─── Prediction cache ─────────────────────────────────────────────────────────

def build_predictions_cache(
    params: TradeSimParams,
    cache_file: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Run NN inference for every test date and persist the raw per-step
    predictions to *cache_file*.

    Cache format — ``dict[pd.Timestamp, dict]`` with keys:
        ``trend_pred`` : float  — mean median prediction over signal_horizon_steps
        ``avg_lo_80``  : float | None  — mean lo-80 bound  (None for MAE models)
        ``avg_hi_80``  : float | None  — mean hi-80 bound  (None for MAE models)

    Parameters
    ----------
    params      : TradeSimParams
    cache_file  : destination path  (default: ``<model_path>/predictions_cache.pkl``)
    force       : if True, rebuild even if cache already exists

    Returns
    -------
    Path   path to the written cache file
    """
    if cache_file is None:
        cache_file = Path(params.model_path) / CACHE_FILENAME

    if cache_file.exists() and not force:
        print(f"Cache already exists at {cache_file}  (pass --force to rebuild)")
        return cache_file

    # load_from_folder() skips init=False fields, so model_storage_folder may
    # not be set when params were restored from a tradesimlog folder.
    if not hasattr(params, "model_storage_folder") or params.model_storage_folder is None:
        params.model_storage_folder = Path(params.model_path) / "model"

    print("Building predictions cache…")
    simdata = TradeSimulData(params)
    MODEL_INPUT_LEN = _load_input_size(params.model_path)

    print(f"Loading model from {params.model_storage_folder}")
    nf = NeuralForecast.load(path=str(params.model_storage_folder.absolute()))
    _silence_lightning()
    for m in nf.models:
        m.trainer_kwargs.update(
            {"enable_progress_bar": False, "enable_model_summary": False, "logger": False}
        )

    predict_ticker  = params.traded_symbol
    df_full         = simdata.get_full_period_data()
    test_dates      = simdata.get_test_dates()
    H               = params.signal_horizon_steps

    cache: Dict[pd.Timestamp, dict] = {}

    for i, today in enumerate(test_dates):
        # Build the same rolling input window as the live simulator
        df_list = []
        for uid in df_full["unique_id"].unique():
            series = df_full[df_full["unique_id"] == uid].sort_values("ds")
            window = series[series["ds"] <= today].tail(MODEL_INPUT_LEN)
            if len(window) > 0:
                df_list.append(window)

        df_step  = pd.concat(df_list).reset_index(drop=True)
        forecast = nf.predict(df=df_step)

        rows_df = forecast.query(f"unique_id == '{predict_ticker}_price'").iloc[:H]

        if "PatchTST" in rows_df.columns:          # MAE-trained model
            trend_pred = float(rows_df["PatchTST"].mean())
            cache[today] = {"trend_pred": trend_pred, "avg_lo_80": None, "avg_hi_80": None}
        else:                                        # MQLoss-trained model
            trend_pred = float(rows_df["PatchTST-median"].mean())
            avg_lo_80  = float(rows_df["PatchTST-lo-80"].mean())
            avg_hi_80  = float(rows_df["PatchTST-hi-80"].mean())
            cache[today] = {
                "trend_pred": trend_pred,
                "avg_lo_80":  avg_lo_80,
                "avg_hi_80":  avg_hi_80,
            }

        print(f"  [{i+1:3d}/{len(test_dates)}] {today.date()} → {trend_pred:+.5f}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(cache))
    print(f"\nCache saved → {cache_file}  ({len(cache)} entries)")
    return cache_file


def load_predictions_cache(cache_file: Path) -> dict:
    """Load a previously built predictions cache from disk."""
    return pickle.loads(cache_file.read_bytes())


# ─── Fast replay loop ─────────────────────────────────────────────────────────

def replay_simulation(
    cache: dict,
    opens: dict,        # {Timestamp: float}  open price per test date
    closes: dict,       # {Timestamp: float}  close price per test date
    test_dates: List,
    FEE: float,
    THRESHOLD: float,
    STOPLOSS_THRESHOLD: float,
    TRAILING_STOP_THRESHOLD: float,
) -> float:
    """Replay the trading loop using cached NN predictions.

    Faithful reproduction of the simulator.py inner loop — no model inference.

    Parameters
    ----------
    cache           : dict from build_predictions_cache
    opens / closes  : plain dicts for O(1) price lookup
    test_dates      : ordered list of pd.Timestamps (test period)
    FEE             : round-trip fee fraction (e.g. 0.0005)
    THRESHOLD       : signal trigger magnitude
    STOPLOSS_THRESHOLD      : negative fraction, e.g. -0.12
    TRAILING_STOP_THRESHOLD : fraction of max_profit to lock in, e.g. 0.8

    Returns
    -------
    float   portfolio yield in percent  (e.g. 12.5 means +12.5 %)
    """
    START_VALUE   = Trader.START_VALUE
    TP_THRESHOLD  = 4.0 * abs(STOPLOSS_THRESHOLD)   # same formula as simulator.py

    portfolio_value          = START_VALUE
    portfolio_value_at_entry = START_VALUE
    current_trade_direction  = 0
    entry_price              = None
    max_profit               = 0.0

    for i, today in enumerate(test_dates):
        entry = cache.get(today)
        if entry is None:
            continue

        trend_pred = entry["trend_pred"]

        # ── Signal from cached prediction ────────────────────────────────────
        signal = 0
        if trend_pred > THRESHOLD:
            signal = 1
        elif trend_pred < -THRESHOLD:
            #signal = -1 
            signal = 0  # disable shorting for now

        # ── Position P&L ─────────────────────────────────────────────────────
        if current_trade_direction != 0:
            close_today     = closes[today]
            position_return = current_trade_direction * (close_today - entry_price) / entry_price

            if position_return > max_profit:
                max_profit = position_return

            # Take-profit
            if position_return > TP_THRESHOLD:
                signal = SIGNAL_TRIGGER_TP
            # Trailing stop
            elif (
                TRAILING_STOP_THRESHOLD
                and max_profit > SIGNAL_TRIGGER_TP / 2          # max_profit > 1.0 (same threshold as original)
                and max_profit != 0
                and position_return / max_profit < TRAILING_STOP_THRESHOLD
            ):
                position_return = max_profit * TRAILING_STOP_THRESHOLD
                signal = SIGNAL_TRIGGER_TP
            # Hard stop-loss
            elif position_return < STOPLOSS_THRESHOLD:
                signal = SIGNAL_TRIGGER_STOPLOSS
        else:
            position_return = 0.0
            max_profit      = 0.0

        # Apply cumulative return against entry snapshot (avoids compounding bug)
        portfolio_value = portfolio_value_at_entry * (1.0 + position_return)

        # ── Position management ───────────────────────────────────────────────
        if signal != 0:
            if signal != current_trade_direction:
                if current_trade_direction != 0 and entry_price is not None:
                    # Exit current position
                    position_return         = 0.0
                    signal                  = 0
                    current_trade_direction = 0
                    max_profit              = 0.0

                portfolio_value          *= (1.0 - FEE)
                portfolio_value_at_entry  = portfolio_value

                if signal != 0 and i + 1 < len(test_dates):
                    # Enter new position at next-day open
                    #next_day                = test_dates[i + 1]
                    #entry_price             = opens[next_day]
                    entry_price     = closes[today]
                    current_trade_direction = signal
                    max_profit              = 0.0
                else:
                    entry_price = None

    # Close any open position at end of simulation
    if current_trade_direction != 0 and entry_price is not None:
        portfolio_value *= (1.0 - FEE)

    return (portfolio_value / START_VALUE - 1.0) * 100.0


# ─── Surface roughness analysis ───────────────────────────────────────────────

def _local_roughness(df: pd.DataFrame, k: int = 8) -> float:
    """Mean k-nearest-neighbour yield std — measures surface jaggedness.

    Parameters are normalised to [0,1] before distance calculation so each
    axis contributes equally.

    Returns
    -------
    float   mean local std (high = jagged / fragile, low = smooth)
    """
    param_cols = ["THRESHOLD", "STOPLOSS_THRESHOLD", "TRAILING_STOP_THRESHOLD"]
    X = df[param_cols].values.astype(float)
    y = df["yield_pct"].values

    # Normalise each column to [0, 1]
    col_min, col_max = X.min(axis=0), X.max(axis=0)
    rng = np.where(col_max > col_min, col_max - col_min, 1.0)
    Xn = (X - col_min) / rng

    local_stds = []
    for i in range(len(Xn)):
        dists = np.linalg.norm(Xn - Xn[i], axis=1)
        nn_idx = np.argsort(dists)[1 : k + 1]   # exclude self
        local_stds.append(y[nn_idx].std())

    return float(np.mean(local_stds))


# ─── Optuna surface study ─────────────────────────────────────────────────────

def run_trading_param_surface_study(
    params: TradeSimParams,
    n_trials: int = 600,
    threshold_range: Tuple[float, float] = (0.001, 0.020),
    stoploss_range:  Tuple[float, float] = (-0.25, -0.02),
    trailing_range:  Tuple[float, float] = (0.30,  0.95),
    output_dir: Optional[Path] = None,
    cache_file: Optional[Path] = None,
    force_cache: bool = False,
) -> Path:
    """Run an Optuna QMC study over the three threshold parameters.

    The sampler is set to ``QMCSampler`` (quasi-Monte Carlo with scrambling)
    which gives uniform space coverage — we are mapping, not optimising.

    Parameters
    ----------
    params          : TradeSimParams  (FEE, trading dates, model_path …)
    n_trials        : number of Optuna trials
    threshold_range : (lo, hi) for THRESHOLD
    stoploss_range  : (lo, hi) for STOPLOSS_THRESHOLD   (both negative)
    trailing_range  : (lo, hi) for TRAILING_STOP_THRESHOLD
    output_dir      : folder for CSV + plots  (default: <model_path>/surface_study)
    cache_file      : explicit cache path  (default: <model_path>/predictions_cache.pkl)
    force_cache     : rebuild cache even if it exists

    Returns
    -------
    pd.DataFrame  one row per completed trial
    """
    if output_dir is None:
        output_dir = Path(params.model_path) / "surface_study"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_file is None:
        cache_file = Path(params.model_path) / CACHE_FILENAME

    # ── 1. Build cache if needed ──────────────────────────────────────────────
    build_predictions_cache(params, cache_file, force=force_cache)

    print(f"\nLoading predictions cache from {cache_file}")
    cache = load_predictions_cache(cache_file)

    # ── 2. Load market prices once ────────────────────────────────────────────
    print("Loading market price data…")
    simdata    = TradeSimulData(params)
    test_dates = simdata.get_test_dates()

    # Convert to plain dicts for fast O(1) lookup in the hot loop
    opens_series  = simdata.get_traded_ticker_opens()
    closes_series = simdata.get_traded_ticker_closings()
    opens  = opens_series.to_dict()
    closes = closes_series.to_dict()

    FEE = params.FEE

    # ── 3. Optuna objective ───────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        thr  = trial.suggest_float("THRESHOLD",               *threshold_range)
        sl   = trial.suggest_float("STOPLOSS_THRESHOLD",      *stoploss_range)
        ts   = trial.suggest_float("TRAILING_STOP_THRESHOLD", *trailing_range)
        return replay_simulation(cache, opens, closes, test_dates, FEE, thr, sl, ts)

    sampler = optuna.samplers.QMCSampler(scramble=True)
    
    study   = optuna.create_study(
        direction="maximize",       # surface map; direction doesn't change coverage
        sampler=sampler,
        storage=f"sqlite:///{output_dir}/db.sqlite3",
        study_name=params.traded_symbol + ":thresh_surface",
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nRunning {n_trials} trials…")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── 4. Collect results ────────────────────────────────────────────────────
    records = [
        {**t.params, "yield_pct": t.value}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    df = pd.DataFrame(records)
    csv_path = output_dir / "surface_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}  ({len(df)} trials)")

    # ── 5. Plots + roughness stats ────────────────────────────────────────────
    plot_surface(df, output_dir)

    return output_dir

# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_surface(df: pd.DataFrame, output_dir: Path) -> None:
    """Render:
    * 3-D scatter coloured by yield
    * Pair-wise 2-D heatmaps (marginal mean over the third axis)
    * Local roughness diagnostic
    """
    if df.empty:
        print("No results to plot.")
        return

    PARAMS   = ["THRESHOLD", "STOPLOSS_THRESHOLD", "TRAILING_STOP_THRESHOLD"]
    YLBL     = "Yield %"
    N_BINS   = 15
    best_row = df.loc[df["yield_pct"].idxmax()]

    # ── 1. 3-D scatter ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(
        df["THRESHOLD"],
        df["STOPLOSS_THRESHOLD"],
        df["TRAILING_STOP_THRESHOLD"],
        c=df["yield_pct"],
        cmap="RdYlGn",
        s=18,
        alpha=0.75,
    )
    plt.colorbar(sc, ax=ax, label=YLBL, shrink=0.6, pad=0.1)
    ax.set_xlabel("THRESHOLD", labelpad=8)
    ax.set_ylabel("STOPLOSS_THRESHOLD", labelpad=8)
    ax.set_zlabel("TRAILING_STOP_THRESHOLD", labelpad=8)
    ax.set_title("Yield surface — 3-D scatter", pad=12)
    fig.tight_layout()
    p3d = output_dir / "surface_3d.png"
    fig.savefig(p3d, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"3-D plot        → {p3d}")

    # ── 2. Pair-wise 2-D heatmaps ─────────────────────────────────────────────
    # For each pair (x, y) marginalise over the third by averaging all trials
    # that fall in each 2-D bin.  This shows the marginal landscape.
    pair_specs = [
        ("THRESHOLD",           "STOPLOSS_THRESHOLD",       "TRAILING_STOP_THRESHOLD"),
        ("THRESHOLD",           "TRAILING_STOP_THRESHOLD",  "STOPLOSS_THRESHOLD"),
        ("STOPLOSS_THRESHOLD",  "TRAILING_STOP_THRESHOLD",  "THRESHOLD"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("Yield surface — marginal heatmaps  (average over 3rd axis)", fontsize=13)

    work = df.copy()
    for ax, (xp, yp, zp) in zip(axes, pair_specs):
        work["_xbin"] = pd.cut(work[xp], bins=N_BINS)
        work["_ybin"] = pd.cut(work[yp], bins=N_BINS)
        grid = work.groupby(["_xbin", "_ybin"], observed=True)["yield_pct"].mean().unstack()

        vmax = max(abs(grid.values[~np.isnan(grid.values)]).max(), 1e-3)
        im   = ax.imshow(
            grid.values,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
            extent=[work[xp].min(), work[xp].max(), work[yp].min(), work[yp].max()],
        )
        plt.colorbar(im, ax=ax, label=YLBL)
        ax.set_xlabel(xp)
        ax.set_ylabel(yp)
        ax.set_title(f"{xp}\nvs {yp}")
        work.drop(columns=["_xbin", "_ybin"], inplace=True)

    fig.tight_layout()
    pslice = output_dir / "surface_slices.png"
    fig.savefig(pslice, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap slices  → {pslice}")

    # ── 3. Yield distribution histogram ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["yield_pct"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0,                    color="red",   linestyle="--", linewidth=1.2, label="0%")
    ax.axvline(df["yield_pct"].mean(), color="gold", linestyle="-",  linewidth=1.5,
               label=f"mean={df['yield_pct'].mean():+.2f}%")
    ax.set_xlabel("Yield %")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of yields across parameter space")
    ax.legend()
    fig.tight_layout()
    phist = output_dir / "surface_hist.png"
    fig.savefig(phist, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Yield histogram → {phist}")

    # ── 4. Roughness / stability diagnostics ─────────────────────────────────
    roughness = _local_roughness(df, k=8)
    frac_pos  = (df["yield_pct"] > 0).mean() * 100.0
    yld       = df["yield_pct"]

    print()
    print("─── Surface roughness indicators ───────────────────────────────────")
    print(f"  Trials             : {len(df)}")
    print(f"  Yield range        : [{yld.min():+.2f}%,  {yld.max():+.2f}%]")
    print(f"  Yield mean ± std   : {yld.mean():+.2f}% ± {yld.std():.2f}%")
    print(f"  Fraction profitable: {frac_pos:.1f}%")
    print(f"  Local k-NN roughness (k=8): {roughness:.3f}%")
    print(f"    → low (<2%)  ≈ smooth surface  (robust edge)")
    print(f"    → high (>5%) ≈ jagged surface  (fragile / overfit)")
    print()
    print(f"  Best trial:")
    print(f"    THRESHOLD              = {best_row['THRESHOLD']:.5f}")
    print(f"    STOPLOSS_THRESHOLD     = {best_row['STOPLOSS_THRESHOLD']:.4f}")
    print(f"    TRAILING_STOP_THRESHOLD= {best_row['TRAILING_STOP_THRESHOLD']:.4f}")
    print(f"    Yield                  = {best_row['yield_pct']:+.2f}%")
    print("─────────────────────────────────────────────────────────────────────")

    # Persist roughness summary alongside the CSV
    summary = {
        "n_trials":          len(df),
        "yield_mean":        round(float(yld.mean()), 4),
        "yield_std":         round(float(yld.std()),  4),
        "yield_min":         round(float(yld.min()),  4),
        "yield_max":         round(float(yld.max()),  4),
        "frac_profitable":   round(frac_pos, 2),
        "local_roughness_k8": round(roughness, 4),
        "best_params": {
            "THRESHOLD":               round(float(best_row["THRESHOLD"]), 6),
            "STOPLOSS_THRESHOLD":      round(float(best_row["STOPLOSS_THRESHOLD"]), 4),
            "TRAILING_STOP_THRESHOLD": round(float(best_row["TRAILING_STOP_THRESHOLD"]), 4),
            "yield_pct":               round(float(best_row["yield_pct"]), 4),
        },
    }
    (output_dir / "surface_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON    → {output_dir / 'surface_summary.json'}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore trading yield surface over threshold parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--symbol",      default="MSFT",  help="Traded symbol")
    src.add_argument(
        "--from-folder",
        metavar="FOLDER",
        help="Load params from an existing tradesimlog run folder",
    )

    p.add_argument("--n-trials",    type=int,   default=600,  help="Optuna trial count")
    p.add_argument("--cache-only",  action="store_true", help="Build cache and exit")
    p.add_argument("--force-cache", action="store_true", help="Rebuild cache even if present")

    # Date / data overrides (only used when --symbol is given)
    p.add_argument("--load-from",    default="2022-01-01", help="load_data_from_date")
    p.add_argument("--trade-start",  default="2024-01-01", help="trading_start")
    p.add_argument("--trade-end",    default="2025-01-01", help="trading_end")
    p.add_argument("--fee",          type=float, default=0.0005, help="Round-trip fee")
    p.add_argument("--horizon",      type=int,   default=3, help="signal_horizon_steps")

    # Parameter space bounds
    p.add_argument("--thr-lo",  type=float, default=0.001, help="THRESHOLD lower bound")
    p.add_argument("--thr-hi",  type=float, default=0.020, help="THRESHOLD upper bound")
    p.add_argument("--sl-lo",   type=float, default=-0.25, help="STOPLOSS lower bound")
    p.add_argument("--sl-hi",   type=float, default=-0.02, help="STOPLOSS upper bound")
    p.add_argument("--ts-lo",   type=float, default=0.30,  help="TRAILING_STOP lower bound")
    p.add_argument("--ts-hi",   type=float, default=0.95,  help="TRAILING_STOP upper bound")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.from_folder:
        params = TradeSimParams.load_from_folder(args.from_folder)
        print(f"Loaded params from {args.from_folder}")
    else:
        symbol = args.symbol
        params = TradeSimParams(
            THRESHOLD=0.004,
            STOPLOSS_THRESHOLD=-0.15,
            TRAILING_STOP_THRESHOLD=0.8,
            FEE=args.fee,
            traded_symbol=symbol,
            tickers=[symbol, "^SPX", "^VIX"],
            load_data_from_date=args.load_from,
            trading_start=args.trade_start,
            trading_end=args.trade_end,
            signal_horizon_steps=args.horizon,
        )

    cache_file = Path(params.model_path) / CACHE_FILENAME

    if args.cache_only:
        build_predictions_cache(params, cache_file, force=args.force_cache)
    else:
        run_trading_param_surface_study(
            params,
            n_trials=args.n_trials,
            threshold_range=(args.thr_lo, args.thr_hi),
            stoploss_range=(args.sl_lo,  args.sl_hi),
            trailing_range=(args.ts_lo,  args.ts_hi),
            cache_file=cache_file,
            force_cache=args.force_cache,
        )
