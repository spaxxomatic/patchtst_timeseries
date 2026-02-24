from datetime import datetime

import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from ticker_data import get_df_for_period, get_ticker
from dataclasses import dataclass, field, asdict, fields as dc_fields
    
class Period(dict):
    pass

def locate_model_folder(traded_symbol, load_data_from_date):
    symbol_clean = traded_symbol.replace('^', '')
    #look for model that was trained with the same symbol and load date , ignoring the trading start date since we want to be able to reuse a model for multiple test periods
    matching_model_folder = None
    checkpoints_path = Path("checkpoints")  
    if checkpoints_path.is_dir():
        for folder in checkpoints_path.iterdir():
            if folder.is_dir() and folder.name.startswith(f"{symbol_clean}_{load_data_from_date}"):
                matching_model_folder = folder.name
                break   
    if matching_model_folder is None:
        print(f"No matching model folder found for symbol {traded_symbol} and load date {load_data_from_date} in checkpoints/")
        return None
    return checkpoints_path / matching_model_folder / "model"

PARAMS_DUMPFILE = "tradesimparams.json"
@dataclass
class TradeSimParams:
    THRESHOLD:float
    STOPLOSS_THRESHOLD:float
    TRAILING_STOP_THRESHOLD:float
    FEE:float
    tickers:list[str]
    load_data_from_date:str
    trading_start:str
    trading_end:str
    traded_symbol:str
    signal_horizon_steps:int = 3
    model_path:str = None  # auto-generated in __post_init__ if not provided
    model_storage_folder:Path = field(init=False)
    logfolder:Path = field(init=False)
    sim_log_file:Path = field(init=False)
    paramsfile:Path = field(init=False)
    sim_stats_results:Path = field(init=False)
    
    def __post_init__(self) -> None:
        if self.model_path is None:
            symbol_clean = self.traded_symbol.replace('^', '')
            self.model_path = str(
                Path("checkpoints") / f"{symbol_clean}_{self.load_data_from_date}_{self.trading_start}"
            )
        self.model_storage_folder = Path(self.model_path) / "model"            
      
        self.logfolder = Path("tradesimlog") / datetime.now().strftime(f"{self.traded_symbol}_%Y%m%d_%H%M%S")
        self.logfolder.mkdir(parents=True, exist_ok=True)
        self.sim_log_file = self.logfolder / "sim_log.csv"
        self.paramsfile =  self.logfolder / PARAMS_DUMPFILE
        self.sim_stats_results = self.logfolder / "simstats.json"
        dump_params_to_json(self, self.paramsfile)
    
    def log_error(self, error):
        errfile:Path = Path(self.logfolder / "error.txt" )
        with errfile.open(mode='a') as f:
            return f.write(error + "\n")        

         
    def is_model_available(self):
        dataset_path = self.model_storage_folder / "dataset.pkl"
        if not dataset_path.is_file():
            print(f"Model path {self.model_storage_folder!s} is invalid, file not found: {dataset_path!s}\n")  
            return False
        return True
    
    @classmethod
    def load_from_folder(cls, logfolder):
        """Restore a TradeSimParams from a previous run folder.

        Uses object.__new__ to bypass __post_init__ so no new timestamped
        folder is created and the original paths are preserved.
        """
        logfolder = Path(logfolder)
        raw_dict = json.loads((logfolder / PARAMS_DUMPFILE).read_text())

        instance = object.__new__(cls)
        for f in dc_fields(cls):
            if f.init and f.name in raw_dict:
                object.__setattr__(instance, f.name, raw_dict[f.name])

        # Restore computed paths pointing at the original folder
        object.__setattr__(instance, 'logfolder',       logfolder)
        object.__setattr__(instance, 'sim_log_file',    logfolder / 'sim_log.csv')
        object.__setattr__(instance, 'paramsfile',      logfolder / PARAMS_DUMPFILE)
        object.__setattr__(instance, 'sim_stats_results', logfolder / 'simstats.json')
        return instance

    def store_chart_results(self, fig) -> Path:
        """Save a matplotlib figure to result.png inside the run folder."""
        import matplotlib.pyplot as plt
        out = self.logfolder / 'result.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Chart saved to {out}")
        return out

    def log_sim_results(self, result):
        Path(self.sim_stats_results).write_text(json.dumps(result, indent=4))

class Trader:
    START_VALUE = 10000.0
    
class Indicator: pass

class NeuralForecastPredictor:
    def __init__(self, nf):
        pass
        
            
class TradeSimulData:
    period_train:Period
    period_test:Period
    df_train:pd.DataFrame
    df_test:pd.DataFrame
    _df_full:pd.DataFrame = field(init=False)
    traded_ticker_raw:pd.DataFrame
    vix_raw:pd.DataFrame
    traded_ticker:str
    params: TradeSimParams
    
    def __init__(self, params:TradeSimParams):
        self.params = params
        load_from_date = params.load_data_from_date        
        self.period_train = Period(start=load_from_date, end=params.trading_start)
        self.period_test = Period(start=params.trading_start, end=params.trading_end)
        self.df_train = get_df_for_period(params.tickers, self.period_train, target_ticker=params.traded_symbol)
        self.df_test  = get_df_for_period(params.tickers, self.period_test,  target_ticker=params.traded_symbol)
        
        #Load actual prices for P&L calculation (need one extra day beyond test end for next-day entry)
        ts = pd.to_datetime(self.period_test['end'])          # Timestamp('2025-10-01 00:00:00')
        later_ts = (ts + pd.Timedelta(days=2)).strftime('%Y-%m-%d') # Timestamp('2025-10-03 00:00:00')
        self.traded_ticker_raw   = get_ticker(params.traded_symbol,
                                              start=self.period_test['start'], end=later_ts)        
        self.vix_raw   = get_ticker('^VIX',
                                              start=self.period_test['start'], end=later_ts)  
        self._df_full =  pd.concat([self.df_train, self.df_test]).sort_values(['unique_id', 'ds']).reset_index(drop=True)

        
    def get_full_period_data(self):
        return self._df_full
    
    def get_traded_ticker_opens(self)->pd.DataFrame:
        return  self.traded_ticker_raw['Open']

    def get_traded_ticker_closings(self)->pd.DataFrame:
        return  self.traded_ticker_raw['Close']
    
    def get_vix_opens(self)->pd.DataFrame:
        return  self.vix_raw['Open']
    
    def get_vix_closings(self)->pd.DataFrame:
        return  self.vix_raw['Close']
    
    def get_test_dates(self)->list:
        return sorted(self.df_test[self.df_test['unique_id'] == f'{self.params.traded_symbol}_price']['ds'].unique())
    
class TradeSimulationStats:
    pass
    
    
def _class_to_dict(cls: type) -> Dict[str, Any]:
    """
    Return a plain dict with *only* the user‑defined attributes of a class.
    Private/special attributes (those starting with '__') are ignored.
    Path objects are converted to strings for JSON serialisation.
    """
    result = {}
    for name, value in vars(cls).items():
        if name.startswith("__") or callable(value):
            continue
        result[name] = str(value) if isinstance(value, Path) else value
    return result
    

def dump_params_to_json(params: TradeSimParams, file_path: str | Path) -> None:

    data = _class_to_dict(params)
    Path(file_path).write_text(json.dumps(data, indent=2))
    print(f"✅ Saved {len(data)} items to {file_path}")

def load_params_from_json(file_path: str | Path, as_namespace: bool = True) -> Any:
    """
    Load the JSON back into a Python object.
    
    * If ``as_namespace`` is True (default) you get a ``SimpleNamespace`` whose
      attributes can be accessed with dot‑notation (e.g. ``params.THRESHOLD``).
    * If ``as_namespace`` is False you get a plain ``dict``.
    """
    raw = json.loads(Path(file_path).read_text())
    if as_namespace:
        return SimpleNamespace(**raw)
    return raw    