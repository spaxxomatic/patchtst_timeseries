from __future__ import annotations
from ticker_data import get_ticker
from IPython.core.inputtransformer2 import tr
from lib.trade import Trade
from matplotlib.pylab import mean

from lib.tradeparams import TradeSimulData, TradeSimParams, Trader, dump_params_to_json, locate_model_folder
from pathlib import Path
import json
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
import logging
import warnings
import os
from lib.visualize import visualize_from_folder

SIGNAL_TRIGGER_STOPLOSS = -2
SIGNAL_TRIGGER_TP       =  2   # trailing take-profit exit
_FALLBACK_INPUT_LEN = 130

def _load_input_size(model_path: str) -> int:
    summary_file = Path(model_path) / 'optuna_summary.json'
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        input_size = summary.get('best_params', {}).get('input_size')
        if input_size is not None:
            print(f"input_size={input_size} loaded from {summary_file}")
            return int(input_size)
    print(f"WARNING: optuna_summary.json not found at {model_path}, falling back to input_size={_FALLBACK_INPUT_LEN}")
    return _FALLBACK_INPUT_LEN


def _load_calibrated_threshold(model_path: str, percentile: str = 'p50', fallback: float = None) -> float:
  #- Reads p50 from optuna_summary.json at startup 
  #- Falls back to params.THRESHOLD if not available                                                                                                                     
  #The p50 means "only trade when today's prediction magnitude is above your own historical median" — principled,
  #adaptive per ticker, and zero lookahead. 
  #You can trivially switch to p25 (more trades) or p75 (fewer, higher conviction) by changing the percentile argument.    
    summary_file = Path(model_path) / 'optuna_summary.json'
    if summary_file.exists():
        cal = json.loads(summary_file.read_text()).get('threshold_calibration')
        if cal and percentile in cal:
            threshold = cal[percentile]
            print(f"Calibrated threshold ({percentile})={threshold:.5f} loaded from {summary_file}")
            return threshold
    if fallback is not None:
        print(f"No calibrated threshold found — using fallback THRESHOLD={fallback}")
    return fallback

_SIM_GENERATED_FILES = ["sim_log.csv", "simstats.json", "result.png", "perf_stats.json"]
_SIM_LOCK_FILE       = "sim.lock"


def run_simulation(tradeSimParams:TradeSimParams):
    lock = tradeSimParams.logfolder / _SIM_LOCK_FILE
    lock.touch()
    success = False
    try:
        _run_simulation_inner(tradeSimParams)
        success = True
    finally:
        lock.unlink(missing_ok=True)
        if not success:
            for fname in _SIM_GENERATED_FILES:
                (tradeSimParams.logfolder / fname).unlink(missing_ok=True)


def _run_simulation_inner(tradeSimParams:TradeSimParams):
    trader = Trader()
    portfolio_value = Trader.START_VALUE
    portfolio_value_at_entry = Trader.START_VALUE
    current_trade_direction = 0
    position_return = 0
    entry_price = None
    max_profit = 0.0          # peak position_return since entry (for trailing stop)
    MODEL_INPUT_LEN = _load_input_size(tradeSimParams.model_path)
    THRESHOLD = _load_calibrated_threshold(
        tradeSimParams.model_path, percentile='p25', fallback=tradeSimParams.THRESHOLD
    )
    THRESHOLD = tradeSimParams.THRESHOLD
    print("Loading data")
    simdata:TradeSimulData = TradeSimulData(tradeSimParams)

    # Load pre-trained model
    print ("Loading model from ", tradeSimParams.model_storage_folder)
    if tradeSimParams.is_model_available():
        nf = NeuralForecast.load(path=str(tradeSimParams.model_storage_folder.absolute()))
    else:
        raise RuntimeError(f"Model not available in {tradeSimParams.model_storage_folder}")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("neuralforecast").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', module=r'pytorch_lightning\..*')
    warnings.filterwarnings('ignore', module=r'lightning\..*')
    for m in nf.models:
        m.trainer_kwargs.update({
            'enable_progress_bar': False,
            'enable_model_summary': False,
            'logger': False,
        })

    predict_ticker = tradeSimParams.traded_symbol
    df_full = simdata.get_full_period_data()
    traded_ticker_open = simdata.get_traded_ticker_opens()
    traded_ticker_close = simdata.get_traded_ticker_closings()
    vix_open = simdata.get_vix_opens()
    vix_close = simdata.get_vix_closings()    
    test_dates = simdata.get_test_dates()
    START_VALUE = trader.START_VALUE
    FEE = tradeSimParams.FEE
    # Several approaches for usign the confidence interval, from simple to more sophisticated:                                                                             
                                                                                                                                        
    #   1. Skip uncertain days (flat when interval is wide)                                                                                
    #   interval_width = cv['PatchTST-hi-80'] - cv['PatchTST-lo-80']                                                                     
    #   threshold = interval_width.median()  # or a fixed value                                                                            
                                                                                                                                        
    #   cv['signal'] = 0  # default: stay flat
    #   cv.loc[(cv['PatchTST-median'] > 0) & (interval_width < threshold), 'signal'] = 1
    #   cv.loc[(cv['PatchTST-median'] < 0) & (interval_width < threshold), 'signal'] = -1

    #   2. Only trade when the interval agrees on direction
    #   # Long only if even the pessimistic bound is positive
    #   cv['signal'] = 0
    #   cv.loc[cv['PatchTST-lo-80'] > 0, 'signal'] = 1   # entire 80% interval is positive
    #   cv.loc[cv['PatchTST-hi-80'] < 0, 'signal'] = -1  # entire 80% interval is negative
    #   This is the most conservative — you only trade when the model is highly confident.

    #   3. Scale position size by confidence
    #   max_width = (cv['PatchTST-hi-80'] - cv['PatchTST-lo-80']).max()
    #   confidence = 1 - (cv['PatchTST-hi-80'] - cv['PatchTST-lo-80']) / max_width  # 0=uncertain, 1=very confident

    #   cv['signal'] = np.where(cv['PatchTST-median'] > 0, 1, -1) * confidence
    #   # signal is now a float between -1 and 1 — use as position size fraction

    #   4. Use the 95% interval for stop-loss sizing
    #   cv['signal'] = np.where(cv['PatchTST-median'] > 0, 1, -1)
    #   # Expected worst case loss if wrong:
    #   cv['risk'] = cv['PatchTST-hi-95'] - cv['PatchTST-lo-95']
    #   # Skip if expected risk exceeds your stop-loss threshold
    #   cv.loc[cv['risk'] > 0.02, 'signal'] = 0
    def get_signal(forecast):
        predictions = [row for _, row in forecast.query(
                    f"unique_id == '{predict_ticker}_price'"
                ).iloc[0:tradeSimParams.signal_horizon_steps].iterrows()]
        
        #preds = forecast.query(f"unique_id == '{predict_ticker}_price'").iloc[0,1,2,3] #get predictions for the next days    
        if 'PatchTST' in predictions[0]:  # MAE-based prediction
            pred_avg = (predictions[0]['PatchTST'] + predictions[1]['PatchTST']) / 2
            signal = 0
            if pred_avg > THRESHOLD:   signal = 1
            elif pred_avg < -THRESHOLD: signal = -1
            return signal, pred_avg, None, None
        else:  # MQLoss-trained net
            return get_signal_from_confidence(predictions)
            

    def get_signal_from_confidence(predictions):
        trend_pred = mean([p['PatchTST-median'] for p in predictions])
        #avg_lo_80  = mean([p['PatchTST-lo-80']  for p in predictions])
        #avg_hi_80  = mean([p['PatchTST-hi-80']  for p in predictions])

        signal = 0
        if trend_pred > float(THRESHOLD):
            signal = 1
        elif trend_pred < -float(THRESHOLD):
            signal = -1

        return signal, trend_pred
    
    def vix_gate(today):
        vix_o = vix_open.loc[today]
        vix_c = vix_close.loc[today]
        if vix_c > 25.0: 
            return 0 
        if (vix_c/vix_o) > 1.15: #15 pct jump today
            return 0
        #if (vix_c/vix_o) > 0.9: #solid drop in vix
        #    return False               
        return 1
        
        
    def get_forecast(today):
        df_list = []
        for uid in df_full['unique_id'].unique():
            series = df_full[df_full['unique_id'] == uid].sort_values('ds')
            window = series[series['ds'] <= today].tail(MODEL_INPUT_LEN)
            if len(window) > 0:
                df_list.append(window)
        df_step = pd.concat(df_list).reset_index(drop=True)
        # Predict next h days
        return nf.predict(df=df_step)

    # Log-Header
    with open(tradeSimParams.sim_log_file, 'w') as f_sim_log_file:
        f_sim_log_file.write("date,pred_momentum,signal,in_market,trade_return,portfolio_value,position_return,vix_gate\n")
    
    TP_THRESHOLD = 4*(abs(tradeSimParams.STOPLOSS_THRESHOLD))
    
    for i, today in enumerate(test_dates):
        # --- Build input window: last MODEL_INPUT_LEN rows per unique_id up to today ---
        
        forecast = get_forecast(today)
        signal, prediction = get_signal(forecast)
        if tradeSimParams.invert_signal:
            signal = -signal
        #vix_gate_signal = vix_gate(today)
        vix_gate_signal = 1
        if vix_gate_signal < 1:            
           signal = 0 # don't trade
        
        # --- P&L with real prices ---
        trade_return = 0.0

        if current_trade_direction != 0:
            close_today = traded_ticker_close.loc[today]
            position_return = current_trade_direction * (close_today - entry_price) / entry_price

            # ── Trailing stop ────────────────────────────────────────────────
            # Track the peak profit reached since entry.
            if position_return > max_profit:
                max_profit = position_return
            ts_thresh = tradeSimParams.TRAILING_STOP_THRESHOLD
            if position_return > TP_THRESHOLD: #take profit 
                signal = SIGNAL_TRIGGER_TP
            elif ts_thresh and max_profit > SIGNAL_TRIGGER_TP/2 and position_return / max_profit < ts_thresh:
                # Price crossed the trailing-stop level intraday — assume we
                # closed there.  Lock in return = max_profit * threshold.
                position_return = max_profit * ts_thresh
                signal = SIGNAL_TRIGGER_TP
            elif vix_gate_signal < 1:
                signal = SIGNAL_TRIGGER_STOPLOSS  # unconditional risk-off exit
            if position_return < tradeSimParams.STOPLOSS_THRESHOLD:
                signal = SIGNAL_TRIGGER_STOPLOSS
        else:
            position_return = 0
            max_profit = 0.0

        this_position_return = position_return
        day_current_signal   = signal
        day_current_pos = current_trade_direction
        # Apply cumulative-from-entry return to the portfolio value *at entry*,
        # not to the rolling portfolio (which would compound a cumulative return
        # on top of itself each day).
        portfolio_value = portfolio_value_at_entry * (1 + this_position_return)        
        
        if signal != 0: #keep position until signal inverts, tp or stoploss
            if signal != current_trade_direction:
                if current_trade_direction != 0 and entry_price is not None:
                    #exit position
                    position_return = 0
                    signal = 0
                    current_trade_direction = 0
                    max_profit = 0.0          # reset peak for the next trade
                portfolio_value *= (1 - FEE) # substract fee
                portfolio_value_at_entry = portfolio_value  # keep snapshot current after every close
                if signal != 0 and i + 1 < len(test_dates):
                    #enter position
                    next_day = test_dates[i + 1]
                    entry_price = traded_ticker_open.loc[next_day]
                    #entry_price = traded_ticker_close.loc[next_day]
                    current_trade_direction = signal
                    max_profit = 0.0          # fresh peak tracking for new position
                else:
                    entry_price = None

        #ci_str = f"CI=[{ci_lo_80:+.5f},{ci_hi_80:+.5f}] " if ci_lo_80 is not None else ""
        tp_str = f"MaxP={max_profit:+.5f} " if day_current_pos != 0 else ""
        print(f"{today.date()} | Pred: {prediction:.5f} | Signal: {day_current_signal} | "
            f"Pos: {current_trade_direction} | {tp_str}Return: {trade_return:+.4f} | Port: {portfolio_value:.2f}")
        #lo_str = f"{ci_lo_80}" if ci_lo_80 is not None else ""
        #hi_str = f"{ci_hi_80}" if ci_hi_80 is not None else ""
        with open(tradeSimParams.sim_log_file, 'a') as f:
            f.write(f"{today.date()},{prediction},{day_current_signal},{day_current_pos},{trade_return},{portfolio_value},{this_position_return},{vix_gate_signal}\n")

    # Close last open position — portfolio_value already reflects the last
    # close price from the final loop iteration; only the exit fee is missing.
    if current_trade_direction != 0 and entry_price is not None:
        last_close = traded_ticker_close.loc[test_dates[-1]]
        final_return = current_trade_direction * (last_close - entry_price) / entry_price
        portfolio_value *= (1 - FEE)
        print(f"\nFinal close | Return: {final_return:+.4f} | Port: {portfolio_value:.2f}$")

    tradeSimParams.log_sim_results( {
        'start_value': START_VALUE, 
        'end_value': portfolio_value , 
        'yield': (portfolio_value/START_VALUE-1)*100
        }
    )
    visualize_from_folder(tradeSimParams.logfolder)
    print(f"\n=== End value: {portfolio_value:.2f}$ (Start: 10000, Yield: {(portfolio_value/10000-1)*100:+.2f}%) ===")
    
if __name__ == "__main__":
    
    for signal_horizon_step in range(3,7):
        traded_symbol = 'BAC'
        params = TradeSimParams(
            THRESHOLD=0.001,
            STOPLOSS_THRESHOLD= -0.3,
            TRAILING_STOP_THRESHOLD= 0.7,        
            FEE= 0.0005,
            traded_symbol = traded_symbol,
            tickers = [traded_symbol, '^SPX', '^VIX'],
            load_data_from_date="2010-01-01",
            trading_start="2025-01-01",
            trading_end="2025-12-24",
            signal_horizon_steps=signal_horizon_step,
            # model_path auto-generated as checkpoints/KO_2020-01-01_2025-01-01
            # override here if needed:
            # model_path="./checkpoints/patchtst_momentum_model_multivar_100days_KO_conf_interval/",
            )
        params.invert_signal = False
        params.model_storage_folder =  locate_model_folder(params.traded_symbol, params.load_data_from_date)
        run_simulation(params)   