tickers = [
  "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","BRK-B","TSLA","UNH",
  "LLY","JPM","XOM","JNJ","V","PG","AVGO","MA","HD","CVX",
  "MRK","ABBV","PEP","COST","ADBE","KO","CSCO","WMT","TMO","MCD",
  "PFE","CRM","BAC","ACN","CMCSA","LIN","NFLX","ABT","ORCL","DHR",
  "AMD","WFC","DIS","TXN","PM","VZ","INTU","COP","CAT","AMGN",
  "NEE","INTC","UNP","LOW","IBM","BMY","SPGI","RTX","HON","BA",
  "UPS","GE","QCOM","AMAT","NKE","PLD","NOW","BKNG","SBUX","MS",
  "ELV","MDT","GS","DE","ADP","LMT","TJX","T","BLK","ISRG",
  "MDLZ","GILD","MMC","AXP","SYK","REGN","VRTX","ETN","LRCX","ADI",
  "SCHW","CVS","ZTS","CI","CB","AMT","SLB","C","BDX","MO",
  "PGR","TMUS","FI","SO","EOG","BSX","CME","EQIX","MU","DUK",
  "PANW","PYPL","AON","SNPS","ITW","KLAC","HUBB","ICE","APD","SHW",
  "CDNS","CSX","NOC","CL","MPC","HUM","FDX","WM","MCK","TGT",
  "ORLY","HCA","FCX","EMR","PXD","MMM","MCO","ROP","CMG","PSX",
  "MAR","PH","APH","GD","USB","NXPI","AJG","NSC","PNC","VLO",
  "F","MSI","GM","TT","EW","CARR","AZO","ADSK","TDG","ANET",
  "SRE","ECL","OXY","PCAR","ADM","MNST","KMB","PSA","CCI","CHTR",
  "MCHP","MSCI","CTAS","WMB","AIG","STZ","HES","NUE","ROST","AFL",
  "KVUE","AEP","IDXX","D","TEL","JCI","MET","GIS","IQV","EXC",
  "WELL","DXCM","HLT","ON","COF","PAYX","TFC","BIIB","O","FTNT",
  "DOW","TRV","DLR","MRNA","CPRT","ODFL","DHI","YUM","SPG","CTSH",
  "AME","BKR","SYY","A","CTVA","CNC","EL","AMP","CEG","HAL",
  "OTIS","ROK","PRU","DD","KMI","VRSK","LHX","DG","FIS","CMI",
  "CSGP","FAST","PPG","GPN","GWW","HSY","BK","XEL","DVN","EA",
  "NEM","ED","URI","VICI","PEG","KR","RSG","LEN","PWR","WST",
  "COR","OKE","VMC","KDP","WBD","ABC","PNR","WRB","ZBRA","ETSY",
  "FTV","RCL","BSY","HPQ","NRG","AAL"
]
from lib.tradeparams import TradeSimParams
from lib.model_trainer import train
for t in tickers:
    params = TradeSimParams(
        THRESHOLD=0.0002,
        STOPLOSS_THRESHOLD=-0.01,
        TRAILING_STOP_THRESHOLD=0.3,
        FEE=0.0005,
        traded_symbol=t,
        tickers=[t, '^SPX' ,'^VIX'],
        load_data_from_date='2015-01-01',
        trading_start='2025-01-01',
        trading_end='2025-11-01',
    )
    if not params.is_model_available():
        try:
            train(params)
        except Exception as ex:
            params.log_error(str(ex))