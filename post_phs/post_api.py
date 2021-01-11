from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo

#def post_breakeven_price():
address = 'https://api.phs.vn/market/utilities.svc/PostBreakevenPrice'

tickers = ['AAA', 'ACB']
breakeven_price = dict()
for ticker in tickers:
    try:
        breakeven_price[ticker] = monte_carlo(ticker=ticker, graph='off')
    except KeyError:
        continue

json_str = json.dumps(breakeven_price)
r = requests.post(url=address, data=json_str,
                  headers={'content-type': 'application/json'})

json.loads(r.json())
