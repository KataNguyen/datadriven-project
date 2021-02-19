from request_phs import *
from breakeven_price.monte_carlo import monte_carlo

class post:

    address_breakeven \
        = 'https://api.phs.vn/market/Utilities.svc/PostBreakevenPrice'

    def __init__(self):
        pass

    def breakeven(self, tickers:list=None, exchanges:list=None) -> pd.DataFrame:

        """
        This method post Monte Carlo model's results in breakeven_price
        sub-project to shared API

        :param tickers: list of tickers, if 'all': run all tickers
        :param exchanges: list of exchanges, if 'all': run all tickers
        :type tickers: list
        :type exchanges: list
        :return: pandas.DataFrame
        """

        start_time = time.time()

        if exchanges == 'all' or tickers == 'all':
            tickers = fa.tickers(exchange='all')
        elif exchanges is not None and exchanges != 'all':
            tickers = []
            for exchange in exchanges:
                tickers += fa.tickers(exchange=exchange)
        elif tickers is not None and tickers != 'all':
            pass

        breakeven_price = dict()
        for ticker in tickers:
            try:
                price = monte_carlo(ticker=ticker)
                breakeven_price[ticker] = adjprice(price)
            except (ValueError, KeyError, IndexError):
                print(ticker + ' cannot be run by Monte Carlo')
                pass

            json_str = {'symbol': json.dumps(breakeven_price)}
            json_str = json.dumps(json_str, separators=(',', ':'))
            r = requests.post(url=self.address_breakeven,
                              data=json_str,
                              headers={'content-type':
                                           'application/json; charset=utf-8'})

            df = pd.DataFrame(json.loads(r.json()['d']),
                              index=['breakeven_price'])
            print(df)
            print('-------------------------')

        print('Finished!')
        print("Total execution time is: %s seconds" %(time.time()-start_time))




post = post()