from attr import dataclass


enum TRADESTATE:
    ST_OPEN,
    ST_CLOSING, 
    ST_CLOSED
    
@dataclass
class Trade:
    start_date:str
    entry:float
    state:TRADESTATE
    
    def get_profit(self):
        
    