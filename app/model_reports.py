"""
Scan checkpoints/ and load every optuna_summary.json into a flat dict.
"""

import json
from pathlib import Path

CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"
PROJECT_ROOT = Path(__file__).parent.parent


def load_model_reports() -> list[dict]:
    """Return one flat dict per checkpoint folder that contains optuna_summary.json."""
    rows = []
    for folder in sorted(CHECKPOINTS.iterdir()):
        if not folder.is_dir():
            continue
        summary_file = folder / "optuna_summary.json"
        if not summary_file.exists():
            continue
        try:
            data = json.loads(summary_file.read_text())
        except Exception:
            continue

        bp = data.get("best_params", {})
        rows.append({
            "folder":               folder.name,
            "symbol":               data.get("traded_symbol", ""),
            "tickers":              ", ".join(data.get("tickers", [])),
            "load_data_from":       data.get("load_data_from", ""),
            "trading_start":        data.get("trading_start", ""),
            "best_hit_rate":        data.get("best_hit_rate"),
            "n_completed_trials":   data.get("n_completed_trials"),
            "runtime_min":          round(data.get("runtime", 0) / 60, 1) if data.get("runtime") else None,
            # best_params
            "input_size":           bp.get("input_size"),
            "patch_len":            bp.get("patch_len"),
            "stride":               bp.get("stride"),
            "encoder_layers":       bp.get("encoder_layers"),
            "n_heads":              bp.get("n_heads"),
            "hidden_size":          bp.get("hidden_size"),
            "linear_hidden_size":   bp.get("linear_hidden_size"),
            "dropout":              bp.get("dropout"),
            "learning_rate":        bp.get("learning_rate"),
            # available charts
            "has_cv_chart":         (folder / "cv_backtest.png").exists(),
            "has_train_chart":      (folder / "training_curves.png").exists(),
        })
    return rows


def get_model_symbols(rows: list[dict]) -> list[str]:
    return sorted({r["symbol"] for r in rows if r["symbol"]})
