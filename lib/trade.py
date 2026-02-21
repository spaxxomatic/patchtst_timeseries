from enum import Enum
from attr import dataclass


class TRADESTATE(Enum):    
    ST_OPEN = 1
    ST_CLOSING = 2
    ST_CLOSED = 3
    
@dataclass
class Trade:
    start_date:str
    entry:float
    state:TRADESTATE
    
    def get_profit(self):
        pass
    


@dataclass
class TradeSimul:
        