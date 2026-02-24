"""
app/regime_monitor.py
─────────────────────
Scan all cv_results.csv files and compute rolling directional accuracy
for the regime change monitor dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── Sector map (GICS, with one editorial exception) ──────────────────────────

SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "ADBE": "Technology", "ADI":  "Technology",
    "AMAT": "Technology", "AMD":  "Technology", "AVGO": "Technology",
    "ACN":  "Technology", "ADP":  "Technology", "CRM":  "Technology",
    "FI":   "Technology", "IBM":  "Technology", "INTC": "Technology",
    "INTU": "Technology", "IT":   "Technology", "LRCX": "Technology",
    "MA":   "Technology", "MSFT": "Technology", "NOW":  "Technology",
    "NVDA": "Technology", "ORCL": "Technology", "QCOM": "Technology",
    "TXN":  "Technology", "V":    "Technology",
    # Communication Services
    "CMCSA": "Communication Services", "DIS":   "Communication Services",
    "GOOG":  "Communication Services", "GOOGL": "Communication Services",
    "META":  "Communication Services", "NFLX":  "Communication Services",
    "T":     "Communication Services", "TMUS":  "Communication Services",
    "VZ":    "Communication Services",
    # Health Care
    "ABBV": "Health Care", "ABT":  "Health Care", "AMGN": "Health Care",
    "BDX":  "Health Care", "BMY":  "Health Care", "BSX":  "Health Care",
    "CI":   "Health Care", "CVS":  "Health Care", "DHR":  "Health Care",
    "ELV":  "Health Care", "GILD": "Health Care", "ISRG": "Health Care",
    "JNJ":  "Health Care", "LLY":  "Health Care", "MDT":  "Health Care",
    "MRK":  "Health Care", "PFE":  "Health Care", "REGN": "Health Care",
    "SYK":  "Health Care", "TMO":  "Health Care", "UNH":  "Health Care",
    "VRTX": "Health Care", "ZTS":  "Health Care",
    # Financials
    "AXP":   "Financials", "BAC":  "Financials", "BLK":  "Financials",
    "BRK-B": "Financials", "C":    "Financials", "CB":   "Financials",
    "CME":   "Financials", "GS":   "Financials", "JPM":  "Financials",
    "MMC":   "Financials", "MS":   "Financials", "PGR":  "Financials",
    "SCHW":  "Financials", "SPGI": "Financials", "WFC":  "Financials",
    # Consumer Staples
    "COST": "Consumer Staples", "MDLZ": "Consumer Staples",
    "MO":   "Consumer Staples", "PEP":  "Consumer Staples",
    "PG":   "Consumer Staples", "PM":   "Consumer Staples",
    "WMT":  "Consumer Staples",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "DECK": "Consumer Discretionary", "HD":   "Consumer Discretionary",
    "MCD":  "Consumer Discretionary", "NKE":  "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX":  "Consumer Discretionary",
    # Industrials
    "BA":  "Industrials", "CAT": "Industrials", "DE":  "Industrials",
    "ETN": "Industrials", "GE":  "Industrials", "HON": "Industrials",
    "LMT": "Industrials", "RTX": "Industrials", "UNP": "Industrials",
    "UPS": "Industrials",
    # Energy
    "COP": "Energy", "CVX": "Energy", "EOG": "Energy",
    "SLB": "Energy", "XOM": "Energy",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "ARE":  "Real Estate",
    "EQIX": "Real Estate", "PLD": "Real Estate",
    # Materials
    "LIN": "Materials",
    # Meme Stocks
    "TSLA": "Meme Stocks",
}

SECTOR_ORDER = [
    "Technology", "Communication Services", "Health Care", "Financials",
    "Consumer Staples", "Consumer Discretionary", "Industrials",
    "Energy", "Utilities", "Real Estate", "Materials", "Meme Stocks", "Unknown",
]

# Accent colours per sector (for the sector header pills)
SECTOR_COLORS: Dict[str, str] = {
    "Technology":              "#3b5bdb",
    "Communication Services":  "#7048e8",
    "Health Care":             "#0ca678",
    "Financials":              "#f59f00",
    "Consumer Staples":        "#74c0fc",
    "Consumer Discretionary":  "#f783ac",
    "Industrials":             "#868e96",
    "Energy":                  "#ff6b35",
    "Utilities":               "#94d82d",
    "Real Estate":             "#63e6be",
    "Materials":               "#a9e34b",
    "Meme Stocks":             "#ff6b6b",
    "Unknown":                 "#495057",
}

CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"
WINDOWS     = [30, 40, 60]   # rolling test_period_days


# ─── Statistics ───────────────────────────────────────────────────────────────

def _wilson_ci(p: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n < 1:
        return 0.0, 1.0
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return float(np.clip(centre - margin, 0.0, 1.0)), float(np.clip(centre + margin, 0.0, 1.0))


def _compute_accuracy(
    predictions:    Dict,
    actual_returns: Dict,
    test_dates:     List,
    n_days:         Optional[int],
) -> Tuple[float, float, float, int]:
    """(accuracy, ci_lo, ci_hi, n_signals) for the last n_days test dates."""
    subset = test_dates[-n_days:] if n_days and n_days < len(test_dates) else test_dates
    n_correct = 0
    n_total   = 0
    for date in subset:
        pred = predictions.get(date)
        if pred is None:
            continue
        actual = actual_returns.get(date)
        if actual is None:
            continue
        if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
            n_correct += 1
        n_total += 1
    if n_total == 0:
        return 0.5, 0.0, 1.0, 0
    acc    = n_correct / n_total
    lo, hi = _wilson_ci(acc, n_total)
    return acc, lo, hi, n_total


def _classify_cell(acc: float, ci_lo: float, ci_hi: float) -> str:
    """'above' | 'below' | 'neutral'  based on whether CI clears 50%."""
    if ci_lo > 0.50:
        return "above"
    if ci_hi < 0.50:
        return "below"
    return "neutral"


def _status(row: dict) -> str:
    """Overall status label for a stock, driven by the 30-day window."""
    cl30   = _classify_cell(row["acc_30d"], row["ci_lo_30d"], row["ci_hi_30d"])
    cl_full = _classify_cell(row["acc_full"], row["ci_lo_full"], row["ci_hi_full"])

    if cl30 == "below":
        return "inverted"        # recent: CI fully below chance  → regime inverted
    if cl30 == "above":
        return "aligned"         # recent: CI fully above chance  → model working
    if cl_full == "below":
        return "inverted_broad"  # full period below, recent uncertain
    if cl_full == "above":
        return "aligned_broad"   # full period above, recent uncertain
    return "noisy"


# ─── Main loader ──────────────────────────────────────────────────────────────

def load_regime_data() -> List[dict]:
    """Return one dict per model that has a cv_results.csv."""
    import pandas as pd

    seen_symbols: set = set()
    rows: List[dict]  = []

    for folder in sorted(CHECKPOINTS.iterdir()):
        if not folder.is_dir():
            continue
        cv_file = folder / "cv_results.csv"
        if not cv_file.exists():
            continue

        # Symbol is the first segment before '_'
        symbol = folder.name.split("_")[0]

        # Deduplicate — keep first (alphabetically earliest folder = oldest model)
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)

        sector = SECTOR_MAP.get(symbol, "Unknown")

        try:
            df = pd.read_csv(cv_file, parse_dates=["ds"])
        except Exception:
            continue

        if "PatchTST-median" not in df.columns or "y" not in df.columns:
            continue

        # One prediction per ds: take the most recent cutoff for each forecast date
        df = df.sort_values(["ds", "cutoff"]).groupby("ds").last().reset_index()
        test_dates    = sorted(df["ds"].tolist())
        preds         = dict(zip(df["ds"], df["PatchTST-median"]))
        actual_returns = dict(zip(df["ds"], df["y"]))

        if not test_dates:
            continue

        # Extract H and trained_len from optuna_summary.json
        H           = 7
        trained_len = None
        summary_file = folder / "optuna_summary.json"
        if summary_file.exists():
            try:
                s = json.loads(summary_file.read_text())
                H           = s.get("config", {}).get("h", H)
                trained_len = s.get("best_params", {}).get("input_size")
            except Exception:
                pass

        last_date = max(test_dates)

        row: dict = {
            "symbol":       symbol,
            "sector":       sector,
            "sector_color": SECTOR_COLORS.get(sector, "#495057"),
            "folder":       folder.name,
            "H":            H,
            "trained_len":  trained_len,
            "n_test_total": len(test_dates),
            "as_of":        last_date.strftime("%Y-%m-%d"),
        }

        for w in WINDOWS:
            if len(test_dates) >= w:
                acc, lo, hi, n = _compute_accuracy(preds, actual_returns, test_dates, w)
            else:
                acc, lo, hi, n = 0.5, 0.0, 1.0, 0
            row[f"acc_{w}d"]   = round(acc,  4)
            row[f"ci_lo_{w}d"] = round(lo,   4)
            row[f"ci_hi_{w}d"] = round(hi,   4)
            row[f"n_{w}d"]     = n
            row[f"class_{w}d"] = _classify_cell(acc, lo, hi)

        acc, lo, hi, n = _compute_accuracy(preds, actual_returns, test_dates, None)
        row["acc_full"]    = round(acc, 4)
        row["ci_lo_full"]  = round(lo,  4)
        row["ci_hi_full"]  = round(hi,  4)
        row["n_full"]      = n
        row["class_full"]  = _classify_cell(acc, lo, hi)
        row["status"]      = _status(row)

        rows.append(row)

    # Sort: sector order → symbol alphabetically
    rank = {s: i for i, s in enumerate(SECTOR_ORDER)}
    rows.sort(key=lambda r: (rank.get(r["sector"], 99), r["symbol"]))
    return rows


def sector_summary(rows: List[dict]) -> List[dict]:
    """Aggregate per-sector: median accuracy + counts per window."""
    from collections import defaultdict
    import statistics

    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[r["sector"]].append(r)

    summaries = []
    for sector in SECTOR_ORDER:
        items = buckets.get(sector, [])
        if not items:
            continue
        s: dict = {
            "sector":       sector,
            "sector_color": SECTOR_COLORS.get(sector, "#495057"),
            "n_stocks":     len(items),
        }
        for w in WINDOWS + ["full"]:
            key = f"acc_{w}d" if w != "full" else "acc_full"
            cls = f"class_{w}d" if w != "full" else "class_full"
            n_key = f"n_{w}d" if w != "full" else "n_full"
            vals = [r[key] for r in items if r.get(n_key, 0) > 0]
            if vals:
                med = statistics.median(vals)
                n_above = sum(1 for r in items if r.get(cls) == "above")
                n_below = sum(1 for r in items if r.get(cls) == "below")
            else:
                med, n_above, n_below = 0.5, 0, 0
            suffix = f"_{w}d" if w != "full" else "_full"
            s[f"median{suffix}"]  = round(med, 4)
            s[f"n_above{suffix}"] = n_above
            s[f"n_below{suffix}"] = n_below
        summaries.append(s)
    return summaries
