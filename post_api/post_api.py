from request_data.request_data import  *
from breakeven_price import *

def post_breakevenprice():
    tickers = request_ticker_list()
    breakeven_price = dict(name='breakeven_price')
    for ticker in tickers:
        breakeven_price[ticker] =

