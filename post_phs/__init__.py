from request_phs import *
from breakeven_price.monte_carlo import monte_carlo

class post:

    address_breakeven \
        = 'https://api.phs.vn/market/Utilities.svc/PostBreakevenPrice'

    def __init__(self):
        pass

    def breakeven(self, tickers='all') -> None:

        """
        This method post Monte Carlo model's results in breakeven_price
        sub-project to shared API

        :param tickers: list of tickers, if 'all': run all tickers
        :return: None
        """

        start_time = time.time()

        breakeven_price = dict()
        if tickers == 'all':
            tickers = []
            for segment in fa.segments:
                tickers += fa.tickers(segment)

        for ticker in tickers:
            try:
                price = monte_carlo(ticker=ticker)
                print(f'Breakeven price of {ticker} is ' + str(price))
                if price < 10000:
                    breakeven_price[ticker] \
                        = '{:.0f}'.format(round(price,-1))
                elif 10000 <= price < 50000:
                    breakeven_price[ticker] \
                        = '{:.0f}'.format(50 * round(price/50))
                else:
                    breakeven_price[ticker] \
                        = '{:.0f}'.format(round(price,-2))
                print(f'Breakeven price of {ticker} is '
                      + str(breakeven_price[ticker]))
            except (ValueError, KeyError):
                print(ticker + ' cannot be run by Monte Carlo')
                pass

        json_str = {'symbol': json.dumps(breakeven_price)}
        json_str = json.dumps(json_str, separators=(',', ':'))
        r = requests.post(url=self.address_breakeven,
                          data=json_str,
                          headers={'content-type':
                                       'application/json; charset=utf-8'})

        print(r)
        print("Total execution time is: %s seconds" %(time.time()-start_time))
        df = pd.DataFrame(json.loads(r.json()['d']), index=['breakeven_price'])
        return df



post = post()