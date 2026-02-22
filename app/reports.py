"""
Scan tradesimlog/ and load every perf_stats.json into a flat dict.
"""

import json
from pathlib import Path

TRADESIMLOG  = Path(__file__).parent.parent / "tradesimlog"
PROJECT_ROOT = Path(__file__).parent.parent


def load_reports() -> list[dict]:
    """Return one flat dict per run folder that has tradesimparams.json.
    Stats columns are None when perf_stats.json is missing (sim not yet run)."""
    rows = []
    for folder in sorted(TRADESIMLOG.iterdir()):
        params_file = folder / "tradesimparams.json"
        if not params_file.exists():
            continue

        # Load params (symbol, dates, model_path)
        try:
            params = json.loads(params_file.read_text())
        except Exception:
            continue

        model_path = params.get("model_path")
        symbol     = params.get("traded_symbol", "")

        # Load stats if available
        data, strat, bh = {}, {}, {}
        stats_file = folder / "perf_stats.json"
        if stats_file.exists():
            try:
                data  = json.loads(stats_file.read_text())
                strat = data.get("strategy", {})
                bh    = data.get("buy_and_hold", {})
            except Exception:
                pass

        cm = data.get("confusion_matrix", {})
        rows.append({
            "run":                   folder.name,
            "symbol":                symbol,
            "trading_start":         params.get("trading_start", ""),
            "trading_end":           params.get("trading_end", ""),
            "horizon_steps":         params.get("signal_horizon_steps", ""),
            "has_results":           bool(data),
            "return_pct":            data.get("strategy_return_pct"),
            "bh_return_pct":         data.get("bh_return_pct"),
            "ann_return_pct":        strat.get("ann_return_pct"),
            "ann_vol_pct":           strat.get("ann_vol_pct"),
            "max_drawdown_pct":      strat.get("max_drawdown_pct"),
            "sharpe":                strat.get("sharpe"),
            "sortino":               strat.get("sortino"),
            "calmar":                strat.get("calmar"),
            "win_pct":               data.get("win_pct"),
            "n_trades":              data.get("n_trades"),
            "n_winners":             data.get("n_winners"),
            "n_losers":              data.get("n_losers"),
            "avg_win_pct":           data.get("avg_win_pct"),
            "avg_loss_pct":          data.get("avg_loss_pct"),
            "signal_changes":        data.get("signal_changes"),
            "long_days":             data.get("long_days"),
            "short_days":            data.get("short_days"),
            "flat_days":             data.get("flat_days"),
            "bh_sharpe":             bh.get("sharpe"),
            "bh_max_drawdown_pct":   bh.get("max_drawdown_pct"),
            "dir_acc_pct":           cm.get("directional_accuracy_pct"),
            "cm_tp":                 cm.get("TP"),
            "cm_fp":                 cm.get("FP"),
            "cm_fn":                 cm.get("FN"),
            "cm_tn":                 cm.get("TN"),
            "model_path":            model_path,
            "has_cv_chart": (
                model_path is not None and
                (PROJECT_ROOT / model_path / "cv_backtest.png").exists()
            ),
        })
    return rows


def get_symbols(rows: list[dict]) -> list[str]:
    return sorted({r["symbol"] for r in rows if r["symbol"]})
