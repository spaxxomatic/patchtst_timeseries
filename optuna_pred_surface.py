"""optuna_pred_surface.py
─────────────────────────────────────────────────────────────────────────────
Explore model DIRECTIONAL ACCURACY surface over three dimensions:

  1. model_input_len  — days of history fed to the model at inference time
                        (fractions of the trained input_size, e.g. 35%…100%)
  2. test_period_days — how many tail-days of the test period to evaluate
                        (30 → full period, in steps)
  3. threshold        — minimum |prediction| to count as a signal

Unlike the trading-yield surface (thresh_surface.py), this study ignores all
trading logic and measures raw directional prediction accuracy:

    dir_acc = fraction of filtered predictions where
              sign(trend_pred) == sign(actual_H_step_return)

Expected accuracy range: 47–55%.  What matters is CONSISTENCY:
  • A flat surface at 52%     → robust, trustworthy edge
  • A spike to 60% in one corner → cherry-pick artefact, not reliable

Workflow
────────
1. Build a multi-input-len cache  (model inference, once per input_len × date).
   Saved to  <model_path>/pred_surface_cache.pkl
2. Run Optuna QMC study — pure in-memory lookups, no further inference.
3. Save results + plots to  <model_path>/pred_surface/

Usage
─────
    python optuna_pred_surface.py --symbol MSFT
    python optuna_pred_surface.py --from-folder tradesimlog/MSFT_20260101_120000
    python optuna_pred_surface.py --symbol MSFT --cache-only
    python optuna_pred_surface.py --symbol MSFT --force-cache
    python optuna_pred_surface.py --symbol MSFT --n-trials 1000
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

from lib.tradeparams import TradeSimParams, TradeSimulData
from neuralforecast import NeuralForecast

# ─── Constants ────────────────────────────────────────────────────────────────
_FALLBACK_INPUT_LEN  = 130
PRED_CACHE_FILENAME  = "pred_surface_cache.pkl"

# Fractions of the trained input_size to pre-compute at cache-build time.
# 1.00 = exact trained window.  Values < 1 test what happens when less
# historical context is available (neuralforecast left-pads shorter series).
_INPUT_LEN_FRACTIONS: List[float] = [0.35, 0.50, 0.65, 0.80, 1.00]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_input_size(model_path: str) -> int:
    summary_file = Path(model_path) / "optuna_summary.json"
    if summary_file.exists():
        try:
            summary   = json.loads(summary_file.read_text())
            input_size = summary.get("best_params", {}).get("input_size")
            if input_size is not None:
                return int(input_size)
        except Exception:
            pass
    return _FALLBACK_INPUT_LEN


def _silence_lightning() -> None:
    for name in ("pytorch_lightning", "neuralforecast", "lightning.pytorch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module=r"pytorch_lightning\..*")
    warnings.filterwarnings("ignore", module=r"lightning\..*")


# ─── Predictions cache (multi-input-len) ──────────────────────────────────────

def build_pred_surface_cache(
    params: TradeSimParams,
    cache_file: Optional[Path] = None,
    force: bool = False,
    input_len_fractions: Optional[List[float]] = None,
) -> Path:
    """Run model inference for several lookback window lengths and persist results.

    For every (input_len, test_date) pair the raw trend prediction is stored.
    Actual H-step price returns (ground truth) are derived from market closes
    and stored once — they are independent of the input window length.

    Cache structure
    ---------------
    dict with keys:
        "input_lens"       : list[int]          precomputed window lengths
        "trained_input_len": int                the model's trained input_size
        "predictions"      : {int: {Timestamp: float}}  per (len, date)
        "actual_H_returns" : {Timestamp: float}  actual cumulative H-day return
        "test_dates"       : list[Timestamp]
        "H"                : int                signal_horizon_steps
    """
    if cache_file is None:
        cache_file = Path(params.model_path) / PRED_CACHE_FILENAME

    if cache_file.exists() and not force:
        print(f"Cache already exists at {cache_file}  (pass force=True to rebuild)")
        return cache_file

    if not hasattr(params, "model_storage_folder") or params.model_storage_folder is None:
        object.__setattr__(params, "model_storage_folder", Path(params.model_path) / "model")

    if input_len_fractions is None:
        input_len_fractions = _INPUT_LEN_FRACTIONS

    trained_input_len = _load_input_size(params.model_path)
    # Clamp minimum to 10, deduplicate, sort
    input_lens = sorted(
        set(max(10, int(f * trained_input_len)) for f in input_len_fractions)
    )

    print("Building prediction surface cache…")
    print(f"  Trained input_size : {trained_input_len}")
    print(f"  Testing input_lens : {input_lens}")

    simdata = TradeSimulData(params)
    H       = params.signal_horizon_steps

    print(f"Loading model from {params.model_storage_folder}")
    nf = NeuralForecast.load(path=str(Path(params.model_storage_folder).absolute()))
    _silence_lightning()
    for m in nf.models:
        m.trainer_kwargs.update(
            {"enable_progress_bar": False, "enable_model_summary": False, "logger": False}
        )

    predict_ticker = params.traded_symbol
    df_full        = simdata.get_full_period_data()
    test_dates     = simdata.get_test_dates()
    closes_series  = simdata.get_traded_ticker_closings()

    # ── Actual H-step returns ─────────────────────────────────────────────────
    # actual_H_return[today] = close[today + H trading_days] / close[today] - 1
    closes_dict = closes_series.to_dict()
    close_dates = sorted(closes_dict.keys())
    close_idx   = {d: i for i, d in enumerate(close_dates)}

    actual_H_returns: Dict[pd.Timestamp, float] = {}
    for today in test_dates:
        idx = close_idx.get(today)
        if idx is None or idx + H >= len(close_dates):
            continue
        future_date = close_dates[idx + H]
        c_today     = closes_dict[today]
        c_future    = closes_dict[future_date]
        if c_today > 0:
            actual_H_returns[today] = (c_future / c_today) - 1.0

    print(f"  Computed {len(actual_H_returns)}/{len(test_dates)} actual {H}-step returns")

    # ── Model inference per input_len ─────────────────────────────────────────
    all_predictions: Dict[int, Dict[pd.Timestamp, float]] = {}
    uids = df_full["unique_id"].unique()

    for input_len in input_lens:
        print(f"\n  ── input_len = {input_len} "
              f"({'trained' if input_len == trained_input_len else f'{input_len/trained_input_len:.0%} of trained'}) ──")
        preds: Dict[pd.Timestamp, float] = {}

        for i, today in enumerate(test_dates):
            df_list = []
            for uid in uids:
                series = df_full[df_full["unique_id"] == uid].sort_values("ds")
                window = series[series["ds"] <= today].tail(input_len)
                if len(window) > 0:
                    df_list.append(window)

            if not df_list:
                continue

            df_step = pd.concat(df_list).reset_index(drop=True)

            try:
                forecast = nf.predict(df=df_step)
                rows_df  = forecast.query(
                    f"unique_id == '{predict_ticker}_price'"
                ).iloc[:H]

                if "PatchTST" in rows_df.columns:          # MAE model
                    trend_pred = float(rows_df["PatchTST"].mean())
                else:                                       # MQLoss model
                    trend_pred = float(rows_df["PatchTST-median"].mean())

                preds[today] = trend_pred
                print(
                    f"    [{i+1:3d}/{len(test_dates)}] {today.date()} → {trend_pred:+.5f}",
                    end="\r",
                )
            except Exception as exc:
                print(f"\n    [{i+1:3d}/{len(test_dates)}] {today.date()} → SKIP ({exc})")

        print(f"\n    {len(preds)}/{len(test_dates)} predictions successful")
        all_predictions[input_len] = preds

    cache = {
        "input_lens":        input_lens,
        "trained_input_len": trained_input_len,
        "predictions":       all_predictions,
        "actual_H_returns":  actual_H_returns,
        "test_dates":        list(test_dates),
        "H":                 H,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(cache))
    total = sum(len(p) for p in all_predictions.values())
    print(f"\nCache saved → {cache_file}  ({total} total predictions)")
    return cache_file


def load_pred_surface_cache(cache_file: Path) -> dict:
    return pickle.loads(cache_file.read_bytes())


# ─── Directional accuracy ─────────────────────────────────────────────────────

def compute_dir_accuracy(
    predictions:    Dict[pd.Timestamp, float],
    actual_returns: Dict[pd.Timestamp, float],
    test_dates:     List[pd.Timestamp],
    test_period_days: int,
    threshold:      float,
) -> Tuple[float, int]:
    """Fraction of filtered predictions where sign(pred) == sign(actual).

    Parameters
    ----------
    predictions       : {date: trend_pred}
    actual_returns    : {date: actual_H_return}
    test_dates        : full ordered list of test dates
    test_period_days  : evaluate only the *last* N dates
    threshold         : skip predictions with |trend_pred| < threshold

    Returns
    -------
    (dir_accuracy, n_signals)
        n_signals = number of predictions that passed the threshold filter
        dir_accuracy defaults to 0.5 when n_signals == 0
    """
    subset = (
        test_dates[-test_period_days:]
        if test_period_days < len(test_dates)
        else test_dates
    )

    n_correct = 0
    n_total   = 0

    for date in subset:
        pred = predictions.get(date)
        if pred is None or abs(pred) < threshold:
            continue
        actual = actual_returns.get(date)
        if actual is None:
            continue
        if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
            n_correct += 1
        n_total += 1

    if n_total == 0:
        return 0.5, 0
    return n_correct / n_total, n_total


# ─── Optuna surface study ─────────────────────────────────────────────────────

def run_pred_surface_study(
    params:              TradeSimParams,
    n_trials:            int = 1000,
    threshold_range:     Tuple[float, float] = (0.0, 0.020),
    output_dir:          Optional[Path] = None,
    cache_file:          Optional[Path] = None,
    force_cache:         bool = False,
    input_len_fractions: Optional[List[float]] = None,
) -> Path:
    """Run Optuna QMC study over (model_input_len, test_period_days, threshold).

    Parameters
    ----------
    params           : TradeSimParams  (defines model_path, signal_horizon_steps…)
    n_trials         : number of QMC trials
    threshold_range  : (lo, hi) for the threshold dimension
    output_dir       : output folder  (default: <model_path>/pred_surface)
    cache_file       : cache path  (default: <model_path>/pred_surface_cache.pkl)
    force_cache      : rebuild cache even if present
    input_len_fractions : fractions of trained input_size to pre-compute

    Returns
    -------
    Path  output directory
    """
    if output_dir is None:
        output_dir = Path(params.model_path) / "pred_surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_file is None:
        cache_file = Path(params.model_path) / PRED_CACHE_FILENAME

    # ── 1. Build / load cache ─────────────────────────────────────────────────
    build_pred_surface_cache(
        params, cache_file,
        force=force_cache,
        input_len_fractions=input_len_fractions,
    )

    print(f"\nLoading prediction surface cache from {cache_file}")
    cache_data = load_pred_surface_cache(cache_file)

    input_lens       = cache_data["input_lens"]
    all_predictions  = cache_data["predictions"]
    actual_H_returns = cache_data["actual_H_returns"]
    test_dates       = cache_data["test_dates"]
    max_test_days    = len(test_dates)

    print(f"  input_lens    : {input_lens}")
    print(f"  test_dates    : {max_test_days} days")
    print(f"  actual returns: {len(actual_H_returns)} days with ground truth")

    # ── 2. Optuna objective ───────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        input_len   = trial.suggest_categorical("model_input_len", input_lens)
        period_days = trial.suggest_int("test_period_days", 30, max_test_days)
        thr         = trial.suggest_float("threshold", *threshold_range)

        predictions = all_predictions.get(input_len, {})
        dir_acc, n_signals = compute_dir_accuracy(
            predictions, actual_H_returns, test_dates, period_days, thr
        )
        trial.set_user_attr("n_signals", n_signals)
        return dir_acc

    sampler = optuna.samplers.QMCSampler(scramble=True)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{output_dir}/db.sqlite3",
        study_name=params.traded_symbol + ":pred_surface",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\nRunning {n_trials} trials (fast — no inference)…")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── 3. Collect + save results ─────────────────────────────────────────────
    records = [
        {
            **t.params,
            "dir_acc":   t.value,
            "n_signals": t.user_attrs.get("n_signals", 0),
        }
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    df = pd.DataFrame(records)
    csv_path = output_dir / "pred_surface_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}  ({len(df)} trials)")

    # ── 4. Plots + summary ────────────────────────────────────────────────────
    plot_pred_surface(df, output_dir, input_lens)

    return output_dir


# ─── Statistics helpers ───────────────────────────────────────────────────────

def _wilson_ci(p: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95 % confidence interval for a proportion.

    More accurate than the normal approximation, especially for small n or
    extreme p.  Returns (lower, upper) clamped to [0, 1].
    """
    if n < 1:
        return 0.0, 1.0
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return float(np.clip(centre - margin, 0.0, 1.0)), float(np.clip(centre + margin, 0.0, 1.0))


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_pred_surface(
    df:         pd.DataFrame,
    output_dir: Path,
    input_lens: Optional[List[int]] = None,
) -> None:
    """Generate four diagnostic plots and a summary JSON."""
    if df.empty:
        print("No results to plot.")
        return

    if input_lens is None:
        input_lens = sorted(df["model_input_len"].unique().tolist())

    N_BINS = 12
    CMAP   = "RdYlGn"
    CHANCE = 0.50

    # ── 1. Pairwise marginal heatmaps ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "Directional Accuracy Surface — marginal heatmaps  (mean over 3rd axis)",
        fontsize=12,
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#141720")

    pair_specs = [
        ("test_period_days", "threshold",        "model_input_len"),
        ("model_input_len",  "threshold",        "test_period_days"),
        ("model_input_len",  "test_period_days", "threshold"),
    ]

    work = df.copy()

    def _bin_col(w: pd.DataFrame, col: str, n: int = N_BINS):
        if w[col].nunique() <= 8:
            return w[col].astype(str)
        return pd.cut(w[col], bins=n)

    for ax, (xp, yp, _) in zip(axes, pair_specs):
        work["_xbin"] = _bin_col(work, xp)
        work["_ybin"] = _bin_col(work, yp)
        grid = work.groupby(["_xbin", "_ybin"], observed=True)["dir_acc"].mean().unstack()

        vals   = grid.values[~np.isnan(grid.values)]
        spread = max(abs(vals.min() - CHANCE), abs(vals.max() - CHANCE), 0.025) if len(vals) else 0.05
        vmin_a = CHANCE - spread
        vmax_a = CHANCE + spread

        im = ax.imshow(
            grid.values,
            aspect="auto",
            origin="lower",
            cmap=CMAP,
            vmin=vmin_a,
            vmax=vmax_a,
        )
        plt.colorbar(im, ax=ax, label="Dir Accuracy")

        def _lbl(v):
            if isinstance(v, pd.Interval):
                return f"{v.mid:.3g}"
            return str(v)

        ax.set_xticks(range(len(grid.columns)))
        ax.set_yticks(range(len(grid.index)))
        ax.set_xticklabels([_lbl(c) for c in grid.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([_lbl(r) for r in grid.index], fontsize=7)
        ax.set_xlabel(xp, labelpad=6, color="#8892b0")
        ax.set_ylabel(yp, labelpad=6, color="#8892b0")
        ax.set_title(f"{xp}  vs  {yp}", color="#cdd5f3")
        ax.tick_params(colors="#8892b0")

        work.drop(columns=["_xbin", "_ybin"], inplace=True)

    fig.tight_layout()
    heatmap_path = output_dir / "pred_surface_heatmaps.png"
    fig.savefig(heatmap_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Heatmaps        → {heatmap_path}")

    # ── 2. Line plots: accuracy vs threshold / vs test period (with 95 % CI) ────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#141720")
    ax2.set_facecolor("#141720")
    fig.suptitle(
        "Directional accuracy — sensitivity to threshold and test-period length  (shaded = 95 % Wilson CI)",
        fontsize=12, color="#e0e0e0",
    )

    colors     = plt.cm.plasma(np.linspace(0.1, 0.9, len(input_lens)))
    max_period = df["test_period_days"].max()

    def _plot_with_ci(ax, mids, grp, color, label):
        """Plot a mean line with a shaded 95 % Wilson CI band."""
        lows, highs = [], []
        for _, row in grp.iterrows():
            lo, hi = _wilson_ci(row["mean"], row["n_sig"])
            lows.append(lo)
            highs.append(hi)
        ax.fill_between(mids, lows, highs, color=color, alpha=0.13, linewidth=0)
        ax.plot(mids, grp["mean"].values, color=color,
                marker="o", markersize=3.5, linewidth=1.5, label=label)

    # (a) Accuracy vs threshold  (full test period)
    full_df = df[df["test_period_days"] >= max_period * 0.85]
    for color, ilen in zip(colors, input_lens):
        sub = full_df[full_df["model_input_len"] == ilen].copy()
        if len(sub) < 5:
            continue
        sub["_tbin"] = pd.cut(sub["threshold"], bins=15)
        grp = sub.groupby("_tbin", observed=True).agg(
            mean=("dir_acc",   "mean"),
            cnt= ("dir_acc",   "count"),
            n_sig=("n_signals","mean"),
        )
        grp = grp[grp["cnt"] >= 2]
        mids = [iv.mid for iv in grp.index]
        _plot_with_ci(ax1, mids, grp, color, label=f"win={ilen}")

    ax1.axhline(CHANCE, color="white", linestyle="--", linewidth=1, alpha=0.4, label="50% chance")
    ax1.set_xlabel("Threshold", color="#8892b0")
    ax1.set_ylabel("Directional Accuracy", color="#8892b0")
    ax1.set_title("Accuracy vs Threshold\n(full test period, by input window)", color="#cdd5f3")
    ax1.legend(fontsize=8, loc="upper left", facecolor="#1e2130", labelcolor="#8892b0")
    ax1.set_ylim(0.30, 0.80)
    ax1.grid(alpha=0.12, color="#2a2f45")
    ax1.tick_params(colors="#8892b0")

    # (b) Accuracy vs test_period_days  (low-threshold region)
    low_thr_df = df[df["threshold"] <= df["threshold"].quantile(0.20)]
    for color, ilen in zip(colors, input_lens):
        sub = low_thr_df[low_thr_df["model_input_len"] == ilen].copy()
        if len(sub) < 5:
            continue
        sub["_dbin"] = pd.cut(sub["test_period_days"], bins=15)
        grp = sub.groupby("_dbin", observed=True).agg(
            mean=("dir_acc",   "mean"),
            cnt= ("dir_acc",   "count"),
            n_sig=("n_signals","mean"),
        )
        grp = grp[grp["cnt"] >= 2]
        mids = [iv.mid for iv in grp.index]
        _plot_with_ci(ax2, mids, grp, color, label=f"win={ilen}")

    ax2.axhline(CHANCE, color="white", linestyle="--", linewidth=1, alpha=0.4, label="50% chance")
    ax2.set_xlabel("Test Period Days", color="#8892b0")
    ax2.set_ylabel("Directional Accuracy", color="#8892b0")
    ax2.set_title("Accuracy vs Test Period Length\n(threshold≈0, by input window)", color="#cdd5f3")
    ax2.legend(fontsize=8, loc="upper left", facecolor="#1e2130", labelcolor="#8892b0")
    ax2.set_ylim(0.30, 0.80)
    ax2.grid(alpha=0.12, color="#2a2f45")
    ax2.tick_params(colors="#8892b0")

    fig.tight_layout()
    lines_path = output_dir / "pred_surface_lines.png"
    fig.savefig(lines_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Line plots      → {lines_path}")

    # ── 3. Distribution histogram ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#141720")
    ax.hist(df["dir_acc"], bins=40, color="#4dabf7", edgecolor="#1e2440", alpha=0.85)
    ax.axvline(CHANCE, color="#ff6b6b", linestyle="--", linewidth=1.8, label="50% (chance)")
    ax.axvline(df["dir_acc"].mean(), color="#fcc419", linestyle="-", linewidth=2.0,
               label=f"mean = {df['dir_acc'].mean():.3f}")
    ax.set_xlabel("Directional Accuracy", color="#8892b0")
    ax.set_ylabel("Count", color="#8892b0")
    ax.set_title("Distribution of directional accuracy across the parameter space",
                 color="#cdd5f3")
    ax.legend(facecolor="#1e2130", labelcolor="#8892b0")
    ax.tick_params(colors="#8892b0")
    ax.grid(alpha=0.10, color="#2a2f45")
    fig.tight_layout()
    hist_path = output_dir / "pred_surface_hist.png"
    fig.savefig(hist_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Histogram       → {hist_path}")

    # ── 4. Accuracy + signal count vs threshold  (trained window, full period) ─
    trained_len = max(input_lens)
    sub = df[
        (df["model_input_len"] == trained_len) &
        (df["test_period_days"] >= max_period * 0.85)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#141720")

    if len(sub) >= 10:
        sub["_tbin"] = pd.cut(sub["threshold"], bins=18)
        grp = sub.groupby("_tbin", observed=True).agg(
            dir_acc=("dir_acc", "mean"),
            n_signals=("n_signals", "mean"),
        ).dropna()
        mids = [iv.mid for iv in grp.index]

        ax_r = ax.twinx()
        ax_r.set_facecolor("#141720")
        ax.plot(mids, grp["dir_acc"].values, color="#4dabf7",
                marker="o", markersize=4.5, linewidth=2, label="Dir accuracy (left)")
        ax_r.plot(mids, grp["n_signals"].values, color="#ff8c42",
                  marker="s", markersize=4.5, linewidth=2, linestyle="--",
                  label="Signal count (right)")
        ax.axhline(CHANCE, color="#ff6b6b", linestyle="--", alpha=0.6)
        ax.set_xlabel("Threshold", color="#8892b0")
        ax.set_ylabel("Directional Accuracy", color="#4dabf7")
        ax_r.set_ylabel("Signal Count", color="#ff8c42")
        ax.set_title(
            f"Accuracy & Signal Count vs Threshold  (full period, input_len={trained_len})",
            color="#cdd5f3",
        )
        ax.set_ylim(0.35, 0.75)
        ax.tick_params(colors="#8892b0")
        ax_r.tick_params(colors="#ff8c42")
        ax.grid(alpha=0.10, color="#2a2f45")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9,
                  facecolor="#1e2130", labelcolor="#8892b0")

    fig.tight_layout()
    cov_path = output_dir / "pred_surface_coverage.png"
    fig.savefig(cov_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Coverage plot   → {cov_path}")

    # ── 5. Summary JSON + console report ─────────────────────────────────────
    acc           = df["dir_acc"]
    best_row      = df.loc[df["dir_acc"].idxmax()]
    frac_above_50 = round(float((acc > 0.50).mean() * 100), 2)
    frac_above_52 = round(float((acc > 0.52).mean() * 100), 2)
    frac_above_55 = round(float((acc > 0.55).mean() * 100), 2)

    summary = {
        "n_trials":         len(df),
        "dir_acc_mean":     round(float(acc.mean()), 4),
        "dir_acc_std":      round(float(acc.std()),  4),
        "dir_acc_min":      round(float(acc.min()),  4),
        "dir_acc_max":      round(float(acc.max()),  4),
        "frac_above_50pct": frac_above_50,
        "frac_above_52pct": frac_above_52,
        "frac_above_55pct": frac_above_55,
        "best_params": {
            "model_input_len":  int(best_row["model_input_len"]),
            "test_period_days": int(best_row["test_period_days"]),
            "threshold":        round(float(best_row["threshold"]), 6),
            "dir_acc":          round(float(best_row["dir_acc"]), 4),
            "n_signals":        int(best_row["n_signals"]),
        },
    }
    summary_path = output_dir / "pred_surface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON    → {summary_path}")

    print()
    print("─── Directional accuracy indicators ─────────────────────────────────────")
    print(f"  Trials             : {len(df)}")
    print(f"  Accuracy range     : [{acc.min():.3f},  {acc.max():.3f}]")
    print(f"  Accuracy mean ± std: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"  Trials > 50%       : {frac_above_50:.1f}%")
    print(f"  Trials > 52%       : {frac_above_52:.1f}%")
    print(f"  Trials > 55%       : {frac_above_55:.1f}%")
    print(f"\n  Best params:")
    print(f"    model_input_len  = {best_row['model_input_len']}")
    print(f"    test_period_days = {best_row['test_period_days']}")
    print(f"    threshold        = {best_row['threshold']:.5f}")
    print(f"    dir_acc          = {best_row['dir_acc']:.4f}  "
          f"(n_signals = {int(best_row['n_signals'])})")
    print("─────────────────────────────────────────────────────────────────────────")


# ─── Replot helper ────────────────────────────────────────────────────────────

def replot_from_existing(output_dir: Path) -> None:
    """Re-generate all plots from an existing pred_surface_results.csv.

    Use this to refresh charts (e.g. after a code update) without re-running
    the full Optuna study or model inference.

    Parameters
    ----------
    output_dir : path to the pred_surface folder
                 (e.g. checkpoints/AAPL_.../pred_surface/)
    """
    csv_path = Path(output_dir) / "pred_surface_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results CSV found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trials from {csv_path}")
    input_lens = sorted(df["model_input_len"].unique().tolist())
    print(f"input_lens: {input_lens}")
    plot_pred_surface(df, Path(output_dir), input_lens)
    print("Done — plots updated.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore directional prediction accuracy surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--symbol",      default="MSFT", help="Traded symbol")
    src.add_argument(
        "--from-folder",
        metavar="FOLDER",
        help="Load params from an existing tradesimlog run folder",
    )

    p.add_argument("--n-trials",    type=int,   default=1000, help="Optuna trial count")
    p.add_argument("--cache-only",  action="store_true", help="Build cache and exit")
    p.add_argument("--force-cache", action="store_true", help="Rebuild cache even if present")
    p.add_argument(
        "--replot",
        metavar="OUTPUT_DIR",
        help="Regenerate plots from an existing pred_surface_results.csv — no inference or Optuna",
    )

    # Date / data overrides (--symbol mode only)
    p.add_argument("--load-from",   default="2022-01-01", help="load_data_from_date")
    p.add_argument("--trade-start", default="2024-01-01", help="trading_start")
    p.add_argument("--trade-end",   default="2025-01-01", help="trading_end")
    p.add_argument("--horizon",     type=int,   default=3,    help="signal_horizon_steps")

    p.add_argument("--thr-lo", type=float, default=0.0,   help="Threshold lower bound")
    p.add_argument("--thr-hi", type=float, default=0.020, help="Threshold upper bound")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Fast path: just regenerate plots from an existing CSV
    if args.replot:
        replot_from_existing(args.replot)
        raise SystemExit(0)

    if args.from_folder:
        params = TradeSimParams.load_from_folder(args.from_folder)
        print(f"Loaded params from {args.from_folder}")
    else:
        symbol = args.symbol
        params = TradeSimParams(
            THRESHOLD=0.004,
            STOPLOSS_THRESHOLD=-0.15,
            TRAILING_STOP_THRESHOLD=0.8,
            FEE=0.0005,
            traded_symbol=symbol,
            tickers=[symbol, "^SPX", "^VIX"],
            load_data_from_date=args.load_from,
            trading_start=args.trade_start,
            trading_end=args.trade_end,
            signal_horizon_steps=args.horizon,
        )

    cache_file = Path(params.model_path) / PRED_CACHE_FILENAME

    if args.cache_only:
        build_pred_surface_cache(params, cache_file, force=args.force_cache)
    else:
        run_pred_surface_study(
            params,
            n_trials=args.n_trials,
            threshold_range=(args.thr_lo, args.thr_hi),
            cache_file=cache_file,
            force_cache=args.force_cache,
        )
