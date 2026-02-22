"""
Trade simulation visualisation.

Public API
----------
plot_sim_results(params)  ->  matplotlib Figure
    Generate the 3-panel backtest chart from a completed simulation run.
    The figure is returned (not saved) so the caller decides where to persist it.

visualize_from_folder(folder)
    Load a TradeSimParams from *folder*, render the chart and save it via
    params.store_chart_results().  Useful for post-hoc inspection of any run.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from ticker_data import get_ticker


def plot_sim_results(params) -> matplotlib.figure.Figure:
    """Render the 3-panel backtest chart for *params*.

    Reads the sim log CSV from params.sim_log_file, fetches the raw price
    series and returns the figure.  Does NOT save or display the figure.

    Parameters
    ----------
    params : TradeSimParams
        A fully populated params object (live or loaded via load_from_folder).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Load simulation log ──────────────────────────────────────────────────
    df_log = pd.read_csv(params.sim_log_file, 
                         parse_dates=['date'], index_col='date')
    df_log.index = pd.to_datetime(df_log.index).tz_localize(None).normalize()

    # ── Fetch raw close prices for the traded symbol ─────────────────────────
    df_price = get_ticker(params.traded_symbol,
                          start=params.trading_start,
                          end=params.trading_end)
    closes = df_price['Close'].squeeze()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes.reindex(df_log.index)

    bh_normalized = closes / closes.iloc[0] * 10000

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 1]})

    # Panel 1 – Portfolio P&L vs Buy & Hold
    ax1 = axes[0]
    ax1.plot(df_log.index, df_log['portfolio_value'],
             label='Strategy PNL', linewidth=2, color='tab:blue')
    ax1.plot(bh_normalized.index, bh_normalized.values,
             label='Buy & Hold', linewidth=1.5, color='tab:gray', alpha=0.7)
    ax1.axhline(10000, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Portfolio Value (€)')
    ax1.set_title(
        f'Backtest: {params.traded_symbol} Strategy vs Buy & Hold'
        f'  [{params.trading_start} → {params.trading_end}]'
    )
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Shade long/short periods
    for j in range(len(df_log) - 1):
        in_market = df_log['in_market'].iloc[j]
        vix_gate = df_log['vix_gate'].iloc[j]
        if in_market == 1:
            ax1.axvspan(df_log.index[j], df_log.index[j + 1], alpha=0.08, color='green')
        elif in_market == -1:
            ax1.axvspan(df_log.index[j], df_log.index[j + 1], alpha=0.08, color='pink')
        if vix_gate > 0:
            ax1.axvspan(df_log.index[j], df_log.index[j + 1], alpha=0.3, color='red')


    # Entry / exit markers on the portfolio curve
    prev_pos = df_log['in_market'].shift(1).fillna(0)
    next_pos = df_log['in_market'].shift(-1).fillna(0)

    entry_long  = df_log[(prev_pos != 1)  & (df_log['in_market'] == 1)]
    entry_short = df_log[(prev_pos != -1) & (df_log['in_market'] == -1)]
    exit_long   = df_log[(df_log['in_market'] == 1)  & (next_pos != 1)]
    exit_short  = df_log[(df_log['in_market'] == -1) & (next_pos != -1)]

    ax1.scatter(entry_long.index,  entry_long['portfolio_value'],
                marker='^', color='lime',   edgecolors='darkgreen',
                s=90, zorder=6, label='Long entry')
    ax1.scatter(entry_short.index, entry_short['portfolio_value'],
                marker='v', color='tomato', edgecolors='darkred',
                s=90, zorder=6, label='Short entry')
    ax1.scatter(exit_long.index,   exit_long['portfolio_value'],
                marker='o', facecolors='none', edgecolors='lime',
                linewidths=1.8, s=70, zorder=6, label='Exit long')
    ax1.scatter(exit_short.index,  exit_short['portfolio_value'],
                marker='o', facecolors='none', edgecolors='tomato',
                linewidths=1.8, s=70, zorder=6, label='Exit short')

    # Panel 2 – Prediction + position return overlay
    ax2 = axes[1]
    bar_colors = [
        'green' if s == 1 else 'pink' if s == -1 else 'red' if s == -2 else 'gray'
        for s in df_log['signal']
    ]
    ax2.bar(df_log.index, df_log['pred_momentum'], color=bar_colors, alpha=0.6, width=1)
    ax2.axhline(params.THRESHOLD,  color='green', linestyle='--', alpha=0.5,
                label=f'Threshold (±{params.THRESHOLD})')
    ax2.axhline(-params.THRESHOLD, color='red',   linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Prediction')
    ax2.set_title('PatchTST prediction and Position Evolution')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax_pos = ax2.twinx()
    color_map = {-2: 'tab:red', -1: 'tab:red', 0: 'tab:gray', 1: 'tab:green', 2: 'tab:green'}
    pos_colors = df_log['in_market'].map(color_map)
    ax_pos.scatter(df_log.index, df_log['position_return'] * 100,
                   c=pos_colors, s=12, zorder=3)
    ax_pos.set_ylabel('Position Return (%)')

    # Panel 3 – Raw price
    ax3 = axes[2]
    ax3.plot(closes.index, closes.values, color='tab:orange', linewidth=1.2)
    ax3.set_ylabel(f'{params.traded_symbol} Price ($)')
    ax3.set_title(f'{params.traded_symbol} Close Price')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Stats helper ─────────────────────────────────────────────────────────────

_TRADING_DAYS = 252  # annualisation factor


def _risk_metrics(values: pd.Series) -> dict:
    """Compute risk/return metrics for a portfolio-value or price series."""
    import numpy as np
    daily_ret = values.pct_change().dropna()
    n_days    = len(values)
    ann_factor = _TRADING_DAYS / n_days

    ann_return  = float((values.iloc[-1] / values.iloc[0]) ** ann_factor - 1)
    ann_vol     = float(daily_ret.std() * np.sqrt(_TRADING_DAYS))

    # Max drawdown
    rolling_max  = values.cummax()
    drawdown     = (values - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Sharpe (risk-free = 0 for simplicity)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation only)
    downside     = daily_ret[daily_ret < 0]
    downside_vol = float(downside.std() * np.sqrt(_TRADING_DAYS)) if len(downside) > 1 else 0.0
    sortino      = ann_return / downside_vol if downside_vol > 0 else 0.0

    # Calmar
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        'ann_return_pct':  round(ann_return * 100, 4),
        'ann_vol_pct':     round(ann_vol    * 100, 4),
        'max_drawdown_pct': round(max_drawdown * 100, 4),
        'sharpe':          round(sharpe,  4),
        'sortino':         round(sortino, 4),
        'calmar':          round(calmar,  4),
    }


def print_sim_stats(params) -> dict:
    """Print and save a full performance summary to ``perf_stats.json``.

    Returns the stats dict.
    """
    import json

    df_log = pd.read_csv(params.sim_log_file, parse_dates=['date'], index_col='date')
    df_log.index = pd.to_datetime(df_log.index).tz_localize(None).normalize()

    df_price = get_ticker(params.traded_symbol,
                          start=params.trading_start,
                          end=params.trading_end)
    closes = df_price['Close'].squeeze()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes.reindex(df_log.index)

    total_return = (df_log['portfolio_value'].iloc[-1] / 10000 - 1) * 100
    bh_return    = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
    n_signals    = int((df_log['signal'].diff().fillna(0) != 0).sum())
    long_days    = int((df_log['in_market'] == 1).sum())
    short_days   = int((df_log['in_market'] == -1).sum())
    flat_days    = int((df_log['in_market'] == 0).sum())

    strat_risk = _risk_metrics(df_log['portfolio_value'])
    bh_risk    = _risk_metrics(closes.dropna())

    # ── Win rate ─────────────────────────────────────────────────────────────
    # A trade ends on the last day it has a non-zero in_market before
    # switching to flat or reversing.  position_return on that day is the
    # cumulative return of the trade from entry.
    next_in_market = df_log['in_market'].shift(-1).fillna(0)
    trade_end_mask = (df_log['in_market'] != 0) & (df_log['in_market'] != next_in_market)
    trade_returns  = df_log.loc[trade_end_mask, 'position_return']
    n_trades   = int(len(trade_returns))
    n_winners  = int((trade_returns > 0).sum())
    n_losers   = int((trade_returns < 0).sum())
    win_pct    = round(n_winners / n_trades * 100, 2) if n_trades > 0 else 0.0
    avg_win    = round(float(trade_returns[trade_returns > 0].mean() * 100), 4) if n_winners > 0 else 0.0
    avg_loss   = round(float(trade_returns[trade_returns < 0].mean() * 100), 4) if n_losers  > 0 else 0.0

    # ── Directional confusion matrix ─────────────────────────────────────────
    # pred_momentum is logged for every day (ungated, pre-threshold signal).
    # actual_next_day_return = next close / today close - 1  (shift closes by -1).
    import numpy as np
    next_close      = closes.shift(-1)
    actual_ret      = (next_close - closes) / closes          # next-day return
    pred_sign       = np.sign(df_log['pred_momentum'])
    actual_sign     = np.sign(actual_ret.reindex(df_log.index))

    # Only evaluate days where neither is exactly 0
    mask = (pred_sign != 0) & (actual_sign != 0) & actual_sign.notna()
    ps   = pred_sign[mask]
    as_  = actual_sign[mask]
    n_cm = int(mask.sum())

    tp = int(((ps ==  1) & (as_ ==  1)).sum())
    fp = int(((ps ==  1) & (as_ == -1)).sum())
    fn = int(((ps == -1) & (as_ ==  1)).sum())
    tn = int(((ps == -1) & (as_ == -1)).sum())
    dir_acc = round((tp + tn) / n_cm * 100, 2) if n_cm > 0 else 0.0

    confusion = {
        'n_days':        n_cm,
        'TP':            tp,
        'FP':            fp,
        'FN':            fn,
        'TN':            tn,
        'directional_accuracy_pct': dir_acc,
    }

    stats = {
        'traded_symbol':   params.traded_symbol,
        'trading_start':   params.trading_start,
        'trading_end':     params.trading_end,
        'signal_horizon_steps': getattr(params, 'signal_horizon_steps', None),
        # ── returns ──
        'strategy_return_pct': round(total_return, 4),
        'bh_return_pct':       round(bh_return, 4),
        # ── activity ──
        'signal_changes': n_signals,
        'long_days':      long_days,
        'short_days':     short_days,
        'flat_days':      flat_days,
        # ── trade statistics ──
        'n_trades':   n_trades,
        'n_winners':  n_winners,
        'n_losers':   n_losers,
        'win_pct':    win_pct,
        'avg_win_pct':  avg_win,
        'avg_loss_pct': avg_loss,
        # ── risk metrics (strategy) ──
        'strategy': strat_risk,
        # ── risk metrics (buy & hold) ──
        'buy_and_hold': bh_risk,
        # ── directional confusion matrix ──
        'confusion_matrix': confusion,
    }

    out = Path(params.logfolder) / 'perf_stats.json'
    out.write_text(json.dumps(stats, indent=2))

    print(f"Strategy Return:   {total_return:+.2f}%   (B&H: {bh_return:+.2f}%)")
    print(f"Ann. Vol:          {strat_risk['ann_vol_pct']:+.2f}%   (B&H: {bh_risk['ann_vol_pct']:+.2f}%)")
    print(f"Max Drawdown:      {strat_risk['max_drawdown_pct']:+.2f}%   (B&H: {bh_risk['max_drawdown_pct']:+.2f}%)")
    print(f"Sharpe:            {strat_risk['sharpe']:.3f}   (B&H: {bh_risk['sharpe']:.3f})")
    print(f"Sortino:           {strat_risk['sortino']:.3f}   (B&H: {bh_risk['sortino']:.3f})")
    print(f"Calmar:            {strat_risk['calmar']:.3f}   (B&H: {bh_risk['calmar']:.3f})")
    print(f"Signal Changes:    {n_signals}")
    print(f"Long/Short/Flat:   {long_days} / {short_days} / {flat_days} days")
    print(f"Trades:            {n_trades}  (W: {n_winners} / L: {n_losers})  Win%: {win_pct:.1f}%")
    print(f"Avg Win / Loss:    {avg_win:+.2f}% / {avg_loss:+.2f}%")
    print(f"Directional Acc:   {dir_acc:.1f}%  (TP={tp} FP={fp} FN={fn} TN={tn}  n={n_cm})")
    print(f"Stats saved to     {out}")

    return stats


# ── Test entry point ──────────────────────────────────────────────────────────

def visualize_from_folder(folder: str | Path) -> None:
    """Load a previous run, render the chart and save result.png to that folder.

    Parameters
    ----------
    folder : str | Path
        Path to a tradesimlog run folder that contains tradesimparams.json
        and sim_log.csv (created by simulator.py).

    Example
    -------
    python -c "from lib.visualize import visualize_from_folder; \
               visualize_from_folder('tradesimlog/KO_20260221_181917')"
    """
    from lib.tradeparams import TradeSimParams
    params = TradeSimParams.load_from_folder(Path(folder))
    print(f"Loaded params from {params.logfolder}")
    fig = plot_sim_results(params)
    params.store_chart_results(fig)
    print_sim_stats(params)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m lib.visualize <run_folder>")
        sys.exit(1)
    visualize_from_folder(sys.argv[1])
