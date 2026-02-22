from __future__ import annotations
from IPython.core.inputtransformer2 import tr
from lib.trade import Trade
from matplotlib.pylab import mean

from lib.tradeparams import TradeSimulData, TradeSimParams, Trader, dump_params_to_json
from pathlib import Path
import json
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
import logging
import os
from lib.visualize import visualize_from_folder

SIGNAL_TRIGGER_STOPLOSS = -2
MODEL_INPUT_LEN = 130 #TODO - load from modelfolder/optuna_summary.json - input_size

def run_simulation(tradeSimParams:TradeSimParams):
    trader = Trader()
    portfolio_value = Trader.START_VALUE
    portfolio_value_at_entry = Trader.START_VALUE
    current_pos = 0
    position_return = 0
    entry_price = None
    print("Loading data")
    simdata:TradeSimulData = TradeSimulData(tradeSimParams)
    # Log-Header
    with open(tradeSimParams.sim_log_file, 'w') as f_sim_log_file:
        f_sim_log_file.write("date,pred_momentum,signal,in_market,trade_return,portfolio_value,position_return\n")

    # Load pre-trained model
    print ("Loading model from ", tradeSimParams.model_storage_folder)
    nf = NeuralForecast.load(path=str(tradeSimParams.model_storage_folder.absolute()))
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("neuralforecast").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    predict_ticker = tradeSimParams.traded_symbol
    df_full = simdata.get_full_period_data()
    traded_ticker_open = simdata.get_traded_ticker_opens()
    traded_ticker_close = simdata.get_traded_ticker_closings()
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
        if 'PatchTST' in predictions[0]: #MAE-based prediction
            pred_avg_over_next_2_days = (predictions[0]['PatchTST'] + predictions[1]['PatchTST'])/2
            # Signal logic
            signal = 0
            if pred_avg_over_next_2_days > params.THRESHOLD:  signal = 1
            elif pred_avg_over_next_2_days < -1*params.THRESHOLD: signal = -1
            return (signal, pred_avg_over_next_2_days)
        else : #mq-loss trained net
            return get_signal_from_confidence(predictions)
            

    def get_signal_from_confidence(predictions):
        signal = 0
        #if p1['PatchTST-lo-80'] > 0: 
        #    signal = 1   # entire 80% interval is positive        
        #if p1['PatchTST-hi-80'] < 0:
        #    signal = -1  # entire 80% interval is negative
        trend_pred = mean([p['PatchTST-median'] for p in predictions])
        if trend_pred > params.THRESHOLD: 
            signal = 1       
        if trend_pred < -1*params.THRESHOLD:
            signal = -1  
        print ([p['PatchTST-median'] for p in predictions])
        # pred_momentum < -1*params.THRESHOLD: signal = -1
        return signal, trend_pred
    print("Start simulation")
    for i, today in enumerate(test_dates):
        # --- Build input window: last MODEL_INPUT_LEN rows per unique_id up to today ---
        df_list = []
        for uid in df_full['unique_id'].unique():
            series = df_full[df_full['unique_id'] == uid].sort_values('ds')
            window = series[series['ds'] <= today].tail(MODEL_INPUT_LEN)
            if len(window) > 0:
                df_list.append(window)

        df_step = pd.concat(df_list).reset_index(drop=True)

        # Predict next h days
        forecast = nf.predict(df=df_step)
        
        (signal, prediction) = get_signal(forecast)
        # --- P&L with real prices ---
        trade_return = 0.0

        if current_pos != 0:
            close_today = traded_ticker_close.loc[today]
            position_return = current_pos * (close_today - entry_price) / entry_price
            if position_return < tradeSimParams.STOPLOSS_THRESHOLD:
                signal = SIGNAL_TRIGGER_STOPLOSS
        else:
            position_return = 0

        this_position_return = position_return
        day_current_signal   = signal
        day_current_pos      = current_pos
        # Apply cumulative-from-entry return to the portfolio value *at entry*,
        # not to the rolling portfolio (which would compound a cumulative return
        # on top of itself each day).
        portfolio_value = portfolio_value_at_entry * (1 + this_position_return)

        if signal != 0: #keep position until signal inverts, tp or stoploss
            if signal != current_pos:
                if current_pos != 0 and entry_price is not None:
                    position_return = 0
                    signal = 0
                    current_pos = 0
                portfolio_value *= (1 - FEE)
                portfolio_value_at_entry = portfolio_value  # keep snapshot current after every close
                if signal != 0 and i + 1 < len(test_dates):
                    next_day = test_dates[i + 1]
                    entry_price = traded_ticker_open.loc[next_day]
                    current_pos = signal
                else:
                    entry_price = None

        print(f"{today.date()} | Pred: {prediction:.5f} | Signal: {signal} | "
            f"Pos: {current_pos} | Return: {trade_return:+.4f} | Port: {portfolio_value:.2f}")
        with open(tradeSimParams.sim_log_file, 'a') as f:
            f.write(f"{today.date()},{prediction},{day_current_signal},{day_current_pos},{trade_return},{portfolio_value},{this_position_return}\n")

    # Close last open position — portfolio_value already reflects the last
    # close price from the final loop iteration; only the exit fee is missing.
    if current_pos != 0 and entry_price is not None:
        last_close = traded_ticker_close.loc[test_dates[-1]]
        final_return = current_pos * (last_close - entry_price) / entry_price
        portfolio_value *= (1 - FEE)
        print(f"\nFinal close | Return: {final_return:+.4f} | Port: {portfolio_value:.2f}€")

    tradeSimParams.log_sim_results( {
        'start_value': START_VALUE, 
        'end_value': portfolio_value , 
        'yield': (portfolio_value/START_VALUE-1)*100
        }
    )
    visualize_from_folder(params.logfolder)
    print(f"\n=== Endwert: {portfolio_value:.2f} (Start: 10000€, Rendite: {(portfolio_value/10000-1)*100:+.2f}%) ===")
    
if __name__ == "__main__":
    for signal_horizon_step in range(1,8):
        params = TradeSimParams(
            THRESHOLD=0.0002,
            STOPLOSS_THRESHOLD= -0.01,
            TRAILING_STOP_THRESHOLD= 0.3,        
            FEE= 0.0005,
            traded_symbol = 'INTC',
            tickers = ['INTC', '^SPX'],
            load_data_from_date="2015-01-01",
            trading_start="2025-01-01",
            trading_end="2025-12-01",
            signal_horizon_steps=signal_horizon_step,
            # model_path auto-generated as checkpoints/KO_2020-01-01_2025-01-01
            # override here if needed:
            # model_path="./checkpoints/patchtst_momentum_model_multivar_100days_KO_conf_interval/",
            )
        
        run_simulation(params)   