def post_breakevenprice():
    tickers = request_ticker_list()
    breakeven_price = dict(name='breakeven_price')
    for ticker in tickers:
        try:
            breakeven_price[ticker] =

