from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo

def post_breakeven_price():
    tickers = ['AAA', 'ABS']
    breakeven_price = dict(name='breakeven_price')
    close_price = dict(name='price_at_run')
    for ticker in tickers:
        try:
            close_price[ticker] = request_latest_close_price(ticker)
            breakeven_price[ticker] = monte_carlo(ticker=ticker, graph='off')
        except KeyError:
            continue
    table = pd.concat([close_price, breakeven_price], axis=1)
    return close_price, breakeven_price, table