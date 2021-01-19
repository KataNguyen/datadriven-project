from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo

def post_breakeven_price():

    address = 'https://api.phs.vn/market/Utilities.svc/PostBreakevenPrice'
    #tickers = request_ticker_list()
    tickers = request_ticker_list()
    breakeven_price = pd.Series(index=tickers, name='price', dtype=float)
    for ticker in breakeven_price.index:
        try:
            price = monte_carlo(ticker=ticker, graph='off')
            breakeven_price.loc[ticker] = price
            if price < 10000:
                breakeven_price.loc[ticker] \
                    = '{:.0f}'.format(round(price,-1))
            elif 10000 <= price < 50000:
                breakeven_price.loc[ticker] \
                    = '{:.0f}'.format(50 * round(price/50))
            else:
                breakeven_price.loc[ticker] \
                    = '{:.0f}'.format(round(price,-2))
        except KeyError:
            continue

    breakeven_price = breakeven_price.astype(dtype='float').to_dict()
    json_str = {'symbol': json.dumps(breakeven_price)}
    json_str = json.dumps(json_str, separators=(',', ':'))
    r = requests.post(url=address,
                      data=json_str,
                      headers={'content-type':
                                   'application/json; charset=utf-8'})

    print(r)


