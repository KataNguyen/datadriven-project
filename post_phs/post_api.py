from request_phs.request_data import *
from breakeven_price.monte_carlo import monte_carlo


def post_breakeven_price(tickers='all') -> None:

    """
    This function post Monte Carlo model's results in breakeven_price
    sub-project to shared API

    :param tickers: list of tickers, if 'all': run all tickers
    :return: None
    """

    start_time = time.time()
    address = 'https://api.phs.vn/market/Utilities.svc/PostBreakevenPrice'

    breakeven_price = dict()
    if tickers == 'all':
        tickers = []
        for segment in request_segment_all():
            tickers += request_ticker(segment)

    for ticker in tickers:
        try:
            price = monte_carlo(ticker=ticker)
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
        except (ValueError, KeyError):
            print(ticker + ' cannot be run by Monte Carlo')
            pass

    json_str = {'symbol': json.dumps(breakeven_price)}
    json_str = json.dumps(json_str, separators=(',', ':'))
    r = requests.post(url=address,
                      data=json_str,
                      headers={'content-type':
                                   'application/json; charset=utf-8'})

    print(r)
    print("Total execution time is: %s seconds" %(time.time()-start_time))
    df = pd.DataFrame(json.loads(r.json()['d']), index=['breakeven_price'])
    return df
