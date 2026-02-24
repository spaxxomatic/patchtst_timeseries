"""
Automated PatchTST model training pipeline.

Public API
----------
train(params, config=None)
    Full pipeline: load data → Optuna hyperparameter search → final model training.
    Results (model + Optuna report) are written to params.model_path.

TrainConfig
    Dataclass with all tunable training/search settings.

Usage example
-------------
from lib.tradeparams import TradeSimParams
from lib.model_trainer import TrainConfig, train

params = TradeSimParams(
    THRESHOLD=0.0002,
    STOPLOSS_THRESHOLD=-0.01,
    TRAILING_STOP_THRESHOLD=0.3,
    FEE=0.0005,
    traded_symbol='KO',
    tickers=['KO', '^SPX'],
    load_data_from_date='2020-01-01',
    trading_start='2025-01-01',
    trading_end='2025-11-01',
)
nf, study = train(params)
"""

import json
import logging
import numpy as np
import optuna
from dataclasses import dataclass
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MQLoss


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All knobs for the Optuna search and final training."""
    h: int = 7                     # forecast horizon (days)

    # Optuna search
    n_trials: int = 50             # max number of Optuna trials
    timeout: int = 7200            # wall-clock limit in seconds
    n_folds: int = 4               # rolling folds per trial
    fold_step: int = 20            # trading days between fold cutoffs
    n_cv_windows: int = 3          # cross_validation n_windows per fold
    optuna_max_steps: int = 500    # training steps per trial

    # Final model
    final_max_steps: int = 500     # training steps for final model
    val_size: int = 50             # validation rows withheld during final fit

    # GPU / data loading
    precision: str = '16-mixed'    # '16-mixed' uses fp16 on Ampere, '32' to disable


# ── Optuna objective ──────────────────────────────────────────────────────────

def _build_model(trial, config: TrainConfig) -> PatchTST:
    return PatchTST(
        h=config.h,
        input_size=trial.suggest_int('input_size', 20, 140, step=10),
        patch_len=trial.suggest_categorical('patch_len', [4, 8, 16]),
        stride=trial.suggest_categorical('stride', [2, 4, 8]),
        encoder_layers=trial.suggest_int('encoder_layers', 1, 5),
        n_heads=trial.suggest_categorical('n_heads', [2, 4, 8]),
        hidden_size=trial.suggest_categorical('hidden_size', [32, 64, 128]),
        linear_hidden_size=trial.suggest_categorical('linear_hidden_size', [32, 64, 128, 256, 512]),
        dropout=trial.suggest_float('dropout', 0.1, 0.5),
        learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        max_steps=config.optuna_max_steps,
        loss=MQLoss(level=[80, 95]),
        val_check_steps=config.optuna_max_steps,  # no mid-trial early stopping
        early_stop_patience_steps=-1,
        accelerator='gpu',
        devices=1,
        precision=config.precision,
        enable_progress_bar=False,
    )


def _fold_hit_rate(nf, df_fold, target_uid, n_windows) -> float:
    """Run cross-validation on one fold and return the confidence-filtered hit rate."""
    cv = nf.cross_validation(df=df_fold, n_windows=n_windows, step_size=1, refit=False)
    cv_target = cv[cv['unique_id'] == target_uid]

    actual_dir = np.sign(cv_target['y'])

    # Only trade when the 80% interval doesn't straddle zero
    confident_bull = cv_target['PatchTST-lo-80'] > 0
    confident_bear = cv_target['PatchTST-hi-80'] < 0
    has_signal = confident_bull | confident_bear

    if has_signal.sum() == 0:
        return 0.5  # no signal → neutral

    pred_dir = np.sign(cv_target['PatchTST-median'])
    return float((actual_dir[has_signal] == pred_dir[has_signal]).mean())


def run_optuna(params, df_train, config: TrainConfig) -> optuna.Study:
    """
    Run the Optuna hyperparameter search.

    Saves optuna_summary.json and optuna_trials.csv to params.model_path.

    Returns
    -------
    optuna.Study
    """
    import torch
    import time
    run_start = time.time()
    torch.set_float32_matmul_precision('high')    
    model_dir = Path(params.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_uid = f'{params.traded_symbol}_price'
    all_dates  = np.sort(df_train['ds'].unique())
    n_total    = len(all_dates)

    _silence_loggers()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = _build_model(trial, config)
        nf    = NeuralForecast(models=[model], freq='D')

        fold_hit_rates = []
        for fold in range(config.n_folds):
            cutoff  = all_dates[n_total - config.fold_step * (config.n_folds - fold)]
            df_fold = df_train[df_train['ds'] <= cutoff]
            hr      = _fold_hit_rate(nf, df_fold, target_uid, config.n_cv_windows)
            fold_hit_rates.append(hr)

            # Report consistency score (mean - std) for pruning decisions
            consistency = float(np.mean(fold_hit_rates) - np.std(fold_hit_rates))
            trial.report(consistency, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        trial.set_user_attr('fold_hit_rates', fold_hit_rates)
        return float(np.mean(fold_hit_rates) - np.std(fold_hit_rates))

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout)
    runtime = int(time.time() - run_start)
    print(f"Optuna ran in {runtime} sec")
    _save_optuna_report(study, params, config, model_dir, runtime)
    return study


# ── Final model training ───────────────────────────────────────────────────────

def train_final_model(params, df_train, best_hyperparams: dict, config: TrainConfig) -> NeuralForecast:
    """
    Train a full PatchTST model with *best_hyperparams*, log metrics, plot training
    curves, and save everything to params.model_path.

    Outputs
    -------
    model_dir/metrics.csv          — raw Lightning training metrics
    model_dir/training_curves.png  — train loss + validation loss plot
    model_dir/<nf save files>      — serialised NeuralForecast model
    """
    import glob
    import shutil
    import torch
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pytorch_lightning.loggers import CSVLogger

    torch.set_float32_matmul_precision('high')
    _silence_loggers()

    model_dir = Path(params.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = model_dir / 'logs'
    pl_logger = CSVLogger(str(log_dir), name='training')

    model = PatchTST(
        h=config.h,
        input_size=best_hyperparams['input_size'],
        patch_len=best_hyperparams['patch_len'],
        stride=best_hyperparams['stride'],
        encoder_layers=best_hyperparams['encoder_layers'],
        n_heads=best_hyperparams['n_heads'],
        hidden_size=best_hyperparams['hidden_size'],
        linear_hidden_size=best_hyperparams['linear_hidden_size'],
        dropout=best_hyperparams['dropout'],
        learning_rate=best_hyperparams['learning_rate'],
        max_steps=config.final_max_steps,
        loss=MQLoss(level=[80, 95]),
        val_check_steps=50,
        accelerator='gpu',
        devices=1,
        precision=config.precision,
        enable_progress_bar=True,
        logger=pl_logger,
    )

    nf = NeuralForecast(models=[model], freq='D')

    print(f"Training final model on {len(df_train)} rows  (val_size={config.val_size})…")
    nf.fit(df=df_train, val_size=config.val_size)
    
    nf.save(path=str(params.model_storage_folder))
    print(f"Model saved to {params.model_storage_folder}")

    # ── Copy metrics.csv to model root and plot training curves ──────────────
    version_dirs = sorted(
        glob.glob(str(log_dir / 'training' / 'version_*')),
        key=lambda p: int(p.rsplit('_', 1)[-1]),
    )
    if version_dirs:
        src_metrics = Path(version_dirs[-1]) / 'metrics.csv'
        if src_metrics.exists():
            dst_metrics = model_dir / 'metrics.csv'
            shutil.copy(src_metrics, dst_metrics)
            _plot_training_curves(dst_metrics, model_dir)

    return nf


def _plot_training_curves(metrics_csv: Path, model_dir: Path) -> None:
    """Read Lightning metrics.csv and save training_curves.png."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(metrics_csv)
    train_m = metrics[['step', 'train_loss_step']].dropna()
    val_m   = metrics[['step', 'valid_loss']].dropna()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(train_m['step'], train_m['train_loss_step'],
            alpha=0.25, color='steelblue', linewidth=1, label='Train loss (step)')
    ax.plot(train_m['step'], train_m['train_loss_step'].rolling(20).mean(),
            color='steelblue', linewidth=2, label='Train loss (trend)')
    ax.scatter(val_m['step'], val_m['valid_loss'],
               color='orange', zorder=5, s=40, label='Validation loss')
    ax.plot(val_m['step'], val_m['valid_loss'],
            color='orange', linestyle='--', alpha=0.6)

    ax.set_title('Training curves')
    ax.set_xlabel('Step')
    ax.set_ylabel('MQLoss')
    ax.legend()
    ax.grid(True, alpha=0.25)

    out = model_dir / 'training_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved to {out}")


# ── CV backtest plot ──────────────────────────────────────────────────────────

def plot_cv_backtest(nf: NeuralForecast, df_train, df_test, params, model_dir: Path) -> None:
    """
    Run walk-forward cross-validation over the test period and save a cumulative
    strategy vs buy-and-hold chart to model_dir/cv_backtest.png.

    One CV window per test day (refit=False) — mirrors the evaluation in
    mutivariant.ipynb.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    symbol    = params.traded_symbol
    target_uid = f'{symbol}_price'
    test_start = df_test['ds'].min()
    n_windows  = len(df_test['ds'].unique())

    print(f"Running cross-validation  ({n_windows} windows over test period)…")
    df_full = (
        pd.concat([df_train, df_test])
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    cv = nf.cross_validation(df=df_full, n_windows=n_windows, step_size=1, refit=False)

    # Keep test period only, one prediction per day (first horizon step per cutoff)
    cv_target = cv[(cv['unique_id'] == target_uid) & (cv['ds'] >= test_start)].copy()
    cv_clean  = cv_target.groupby('cutoff').first().reset_index()

    cv_clean['signal']    = np.where(cv_clean['PatchTST-median'] > 0, 1, -1)
    cv_clean['strat_ret'] = cv_clean['signal'] * cv_clean['y']
    cv_clean['cum_strat'] = cv_clean['strat_ret'].cumsum().apply(np.exp)
    cv_clean['cum_market'] = cv_clean['y'].cumsum().apply(np.exp)

    # Save CV results CSV
    cv_clean.to_csv(model_dir / 'cv_results.csv', index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cv_clean['ds'], cv_clean['cum_market'],
            label='Market (Buy & Hold)', color='gray', alpha=0.6, linewidth=1.5)
    ax.plot(cv_clean['ds'], cv_clean['cum_strat'],
            label=f'{symbol} PatchTST strategy', color='steelblue', linewidth=2)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)

    final_strat  = cv_clean['cum_strat'].iloc[-1]
    final_market = cv_clean['cum_market'].iloc[-1]
    ax.set_title(
        f'CV Backtest — {symbol}  [{params.trading_start} → {params.trading_end}]\n'
        f'Strategy: {(final_strat - 1)*100:+.1f}%    B&H: {(final_market - 1)*100:+.1f}%'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative return (normalised)')
    ax.legend()
    ax.grid(True, alpha=0.25)

    out = model_dir / 'cv_backtest.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"CV backtest chart saved to {out}")


# ── Main entry point ──────────────────────────────────────────────────────────

def train(params, config: TrainConfig = None):
    """
    Full pipeline: load data → Optuna search → final model training → CV backtest.

    Parameters
    ----------
    params : TradeSimParams
    config : TrainConfig, optional.  Defaults are used if None.

    Returns
    -------
    (NeuralForecast, optuna.Study)
    """
    from ticker_data import get_df_for_period

    if config is None:
        config = TrainConfig()

    model_dir = Path(params.model_path)

    print(f"Loading data for {params.traded_symbol}…")
    df_train = get_df_for_period(
        params.tickers,
        {'start': params.load_data_from_date, 'end': params.trading_start},
        target_ticker=params.traded_symbol,
    )
    df_test = get_df_for_period(
        params.tickers,
        {'start': params.trading_start, 'end': params.trading_end},
        target_ticker=params.traded_symbol,
    )
    print(f"  Train: {len(df_train['ds'].unique())} days  |  "
          f"Test: {len(df_test['ds'].unique())} days  |  "
          f"{df_train['unique_id'].nunique()} series")

    print(f"\nRunning Optuna  (trials={config.n_trials}, "
          f"folds={config.n_folds}, timeout={config.timeout}s)…")
    study = run_optuna(params, df_train, config)

    print(f"\nBest hit rate : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")

    print("\nTraining final model…")
    nf = train_final_model(params, df_train, study.best_params, config)

    print("\nCalibrating threshold…")
    calibrate_threshold(nf, df_train, params, config, model_dir)

    print("\nGenerating CV backtest…")
    plot_cv_backtest(nf, df_train, df_test, params, model_dir)

    return nf, study


# ── Helpers ───────────────────────────────────────────────────────────────────

def _silence_loggers():
    for name in ("pytorch_lightning", "neuralforecast", "lightning.pytorch"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _save_optuna_report(study: optuna.Study, params, config: TrainConfig, model_dir: Path,
                        runtime: int, threshold_calibration: dict = None):
    import pandas as pd

    # Extract per-fold hit rates from the best trial's user attributes
    best_fold_hrs = study.best_trial.user_attrs.get('fold_hit_rates', [])
    if best_fold_hrs:
        hrs = np.array(best_fold_hrs)
        fold_kpis = {
            'fold_hit_rates':         [round(float(h), 4) for h in hrs],
            'fold_hr_mean':           round(float(hrs.mean()), 4),
            'fold_hr_std':            round(float(hrs.std()),  4),
            'fold_worst_hr':          round(float(hrs.min()),  4),
            'n_folds_above_chance':   int((hrs > 0.5).sum()),
            'fold_consistency_score': round(float(hrs.mean() - hrs.std()), 4),
        }
    else:
        fold_kpis = {}

    summary = {
        'traded_symbol':          params.traded_symbol,
        'tickers':                params.tickers,
        'load_data_from':         params.load_data_from_date,
        'trading_start':          params.trading_start,
        'best_consistency_score': round(study.best_value, 6),
        'best_params':            study.best_params,
        'runtime':                runtime,
        'n_completed_trials': len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        'config': {k: v for k, v in config.__dict__.items()},
        **fold_kpis,
    }
    if threshold_calibration is not None:
        summary['threshold_calibration'] = threshold_calibration

    (model_dir / 'optuna_summary.json').write_text(json.dumps(summary, indent=2))

    df_trials = study.trials_dataframe()
    df_trials.sort_values('value', ascending=False).to_csv(
        model_dir / 'optuna_trials.csv', index=False
    )
    print(f"Optuna report saved to {model_dir}")


def calibrate_threshold(nf: NeuralForecast, df_train, params, config: TrainConfig,
                        model_dir: Path) -> dict:
    """
    Run cross-validation over the last val_size days of df_train (the held-out
    validation window — never seen by the final model during training) and compute
    percentiles of |median| predictions.

    Returns a dict with p25, p50, p75 keys — honest, lookahead-free thresholds.
    Saved into optuna_summary.json under 'threshold_calibration'.
    """
    target_uid = f'{params.traded_symbol}_price'

    print(f"Calibrating threshold over last {config.val_size} val days…")
    cv = nf.cross_validation(
        df=df_train,
        n_windows=config.val_size,
        step_size=1,
        refit=False,
    )

    # One prediction per day — first horizon step per cutoff
    cv_target = cv[cv['unique_id'] == target_uid]
    cv_daily  = cv_target.groupby('cutoff').first().reset_index()

    abs_medians = cv_daily['PatchTST-median'].abs()

    calibration = {
        'p25': round(float(np.percentile(abs_medians, 25)), 6),
        'p50': round(float(np.percentile(abs_medians, 50)), 6),
        'p75': round(float(np.percentile(abs_medians, 75)), 6),
    }
    print(f"  p25={calibration['p25']:.5f}  p50={calibration['p50']:.5f}  "
          f"p75={calibration['p75']:.5f}")

    # Patch the already-written optuna_summary.json
    summary_path = model_dir / 'optuna_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['threshold_calibration'] = calibration
    summary_path.write_text(json.dumps(summary, indent=2))

    return calibration


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    from lib.tradeparams import TradeSimParams
    traded_symbol='^RUT'
    params = TradeSimParams(
        THRESHOLD=0.0002,
        STOPLOSS_THRESHOLD=-0.01,
        TRAILING_STOP_THRESHOLD=0.3,
        FEE=0.0005,
        traded_symbol=traded_symbol,
        tickers=[traded_symbol, '^SPX' ,'^VIX'],
        load_data_from_date='2010-01-01',
        trading_start='2025-01-01',
        trading_end='2025-11-01',
    )
    train(params)
