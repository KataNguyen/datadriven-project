from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo

def post_breakeven_price():

    address = 'https://api.phs.vn/market/Utilities.svc/PostBreakevenPrice'
    tickers = request_ticker()
    breakeven_price = dict(index=tickers, name='price', dtype='int32')
    for ticker in tickers:
        price = monte_carlo(ticker=ticker, graph='off')
        breakeven_price[ticker] = price
        if price < 10000:
            breakeven_price[ticker] \
                = '{:.0f}'.format(round(price,-1))
        elif 10000 <= price < 50000:
            breakeven_price[ticker] \
                = '{:.0f}'.format(50 * round(price/50))
        else:
            breakeven_price[ticker] \
                = '{:.0f}'.format(round(price,-2))

    json_str = {'symbol': json.dumps(breakeven_price)}
    json_str = json.dumps(json_str, separators=(',', ':'))
    r = requests.post(url=address,
                      data=json_str,
                      headers={'content-type':
                                   'application/json; charset=utf-8'})
    print(r)

