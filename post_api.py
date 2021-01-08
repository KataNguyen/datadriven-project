from request_data import *
from breakeven_price import *

def post_breakevenprice():
    tickers = request_ticker_list()
    breakeven_price = dict(name='breakeven_price')
    for ticker in tickers:
        try:
            breakeven_price[ticker] = monte_carlo(ticker=ticker, graph='off')
        except KeyError:
            continue
