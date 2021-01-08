from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo

database_path = join(os.getcwd(),'database')

def post_breakeven_price():
    tickers = request_ticker_list()
    breakeven_price = dict(name='breakeven_price')
    for ticker in tickers:
        try:
            breakeven_price[ticker] = monte_carlo(ticker=ticker, graph='off')
        except KeyError:
            continue
    return breakeven_price