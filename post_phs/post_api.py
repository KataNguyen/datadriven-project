from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo

def post_breakeven_price():

    address = 'https://api.phs.vn/market/Utilities.svc/PostBreakevenPrice'
    #tickers = request_ticker_list()
    tickers = ['AAA', 'ACB', 'HBC', 'CTD', 'VCB', 'TCB']
    breakeven_price = pd.Series(index=tickers, name='price', dtype=float)
    for ticker in breakeven_price.index:
        try:
            price = monte_carlo(ticker=ticker, graph='off')
            if price < 10000:
                breakeven_price.loc[ticker] \
                    = '{:,.0f}'.format(round(price,-1))
            elif 10000 <= price < 50000:
                breakeven_price.loc[ticker] \
                    = '{:,.0f}'.format(50 * round(price/50))
            else:
                breakeven_price.loc[ticker] \
                    = '{:,.0f}'.format(round(price,-2))
        except KeyError:
            continue

    str(breakeven_price.to_dict())
    json_str = json.dumps({'symbol': breakeven_price.to_json()})

    r = requests.post(url=address,
                      data=json_str,
                      headers={'content-type': 'application/json'})

    result = pd.DataFrame(json.loads(r.json()['d']))


