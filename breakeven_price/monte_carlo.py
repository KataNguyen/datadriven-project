from function_phs import *
from request_phs import *
import scipy as sc
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as transforms

destination_dir \
    = join(dirname(realpath(__file__)), 'result')


q_upper = 0.95
q_lower = 0

def monte_carlo(ticker:str, hdays=252, pdays=66, alpha=0.05,
                simulation=100000, savefigure:bool=True,
                fulloutput:bool=False, seed=1):

    start_time = time.time()

    global destination_dir

    def graph_ticker():

        fig, ax = plt.subplots(1, 2, figsize=(12,5.5),
                               sharey=True,
                               gridspec_kw={'width_ratios':[2,1]})
        plt.subplots_adjust(left=0.05, right=0.95,
                            bottom=0.05, top=0.95,
                            wspace=0.01)

        ax[0].set_ylim(np.round(np.min([df_historical['close'].min()*0.95,
                                        np.quantile(df_simulated_price.values,
                                                    0.00)])*0.95, -2),
                       np.round(np.max([df_historical['close'].max()*1.05,
                                        np.quantile(df_simulated_price.values,
                                                    0.95)])*1.05,-2))

        doublediff1 = np.diff(np.sign(np.diff(df_historical['close']
                                              .values)))
        peak_locations = np.where(doublediff1 == -2)[0] + 1 \
                         + df_historical.index[0]
        doublediff2 \
            = np.diff(np.sign(np.diff(-1*df_historical['close'].values)))
        trough_locations = np.where(doublediff2 == -2)[0] + 1 + \
                           df_historical.index[0]
        ax[0].scatter(df_historical['trading_date'][peak_locations],
                      df_historical['close'][peak_locations],
                      marker='^', color='tab:green',
                      s=10, label='Peak')
        ax[0].scatter(df_historical['trading_date'][trough_locations],
                      df_historical['close'][trough_locations],
                      marker='v', color='tab:red',
                      s=10, label='Trough')

        ax[0].plot(df_historical['trading_date'], df_historical['close'],
                   color='midnightblue', alpha=0.9, lw=1.5, label=f'{ticker}')
        ax[0].set_ylabel('Stock Price', fontsize=12)
        ax[0].xaxis.set_major_formatter(DateFormatter('%m/%Y'))
        ax[0].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(priceKformat))
        plt.xticks(rotation=15)
        plt.yticks(fontsize=11)
        ax[0].grid(axis='both', alpha=0.3)

        ax[0].plot(connect, color='tab:red', lw=2, alpha=0.5)
        ax[0].plot(ubound, color='tab:red', lw=2, alpha=0.5)
        ax[0].plot(dbound, color='tab:red', lw=2, alpha=0.5,
                   label='Best Case/ Worst Case')
        ax[0].legend(loc='upper right', fontsize=8)

        ax[0].fill_between(pro_days, ubound.iloc[:,0], dbound.iloc[:,0],
                           color='green', alpha=0.15)
        ax[0].axhline(breakeven_price, ls='--', lw=0.6, color='red')

        ax[0].annotate('Breakeven Price: ' + adjprice(breakeven_price),
                       xy=(0.65,breakeven_price),
                       xycoords=transforms
                       .blended_transform_factory(ax[0].transAxes,
                                                  ax[0].transData),
                       xytext=(0,2), textcoords='offset pixels',
                       ha='left', va='bottom', fontsize=8, color='red')

        ax[0].yaxis\
            .set_major_formatter(FuncFormatter(priceKformat))
        ax[0].annotate('Worst Case over \n'
                       + f'{int(simulation):,d}' + ' Simulations: '
                       + f'{round((breakeven_price/price_t-1)*100, 2):,} % \n'
                       + f'Trading Days: {pdays} \n'
                       + 'Breakeven Price: ' + adjprice(breakeven_price),
                       xy=(0.77, 1.01),
                       xycoords=ax[0].transAxes,
                       ha='left', va='bottom', fontsize=7)

        sns.histplot(y=last_price, ax=ax[1], bins=200,
                     legend=False, color='tab:blue', stat='density')
        ax[1].set_xlabel('Density')
        ax[1].set_ylabel('Stock Price')
        ax[1].axhline(breakeven_price, ls='--', linewidth=0.5, color='red')

        ax[1].annotate('Breakeven Price: ' + adjprice(breakeven_price),
                       xy=(0.57,breakeven_price),
                       xycoords=transforms.blended_transform_factory
                       (ax[1].transAxes,ax[1].transData),
                       xytext=(0,2), textcoords='offset pixels',
                       ha='left', va='bottom', fontsize=8, color='red')

        ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
        ax[1].yaxis.set_major_formatter(FuncFormatter(priceKformat))

        ax[1].grid(axis='both', alpha=0.1)

        if savefigure is True:
            plt.savefig(join(destination_dir,f'{ticker}.png'),
                        bbox_inches='tight')

        return

    # customize display for numpy and pandas
    np.set_printoptions(linewidth=np.inf, precision=0,
                        suppress=True, threshold=int(10e10))
    pd.set_option("display.max_rows", sys.maxsize,
                  "display.max_column", sys.maxsize,
                  'display.width', None,
                  'display.max_colwidth', 20)

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    def Bday(bdays=int):
        d = 0
        bday = datetime.now()
        while d<=bdays:
            bday -= timedelta(days=1)
            d += 1
            while bday.weekday() in holidays.WEEKEND or bday in holidays.VN():
                bday -= timedelta(days=1)
        return bday.strftime("%Y-%m-%d")

    df = ta.hist(ticker, Bday(hdays), today)

    # cleaning data
    df['trading_date'] \
        = pd.to_datetime(df['trading_date'])
    df['close'].loc[df['close'] == 0] \
        = df['ref'].loc[df['close'] == 0]
    df['ref'].loc[df['ref'] == 0] \
        = df['close'].loc[df['ref'] == 0]
    df['change_percent'] \
        = df['close'] / df['ref'] - 1
    df['logr'] \
        = np.log(1 + df['change_percent'])
    df['change_logr'] = 0
    for i in range(1, len(df.index)):
        df['change_logr'].iloc[i] \
            = df['logr'].iloc[i] - df['logr'].iloc[i-1]
    df['change_logr'].iloc[0] = df['change_logr'].iloc[1]
    df['close'] = df['close'] * 1000
    df.drop(columns=['ref', 'high', 'low',
                     'change', 'change_percent',
                     'total_volume', 'total_value'], inplace=True)


    # D'Agostino-Pearson test for log return model
    stat_logr, p_logr = sc.stats.normaltest(df['logr'], nan_policy='omit')
    print(f'p_logr of {ticker} is', p_logr)

    # D'Agostino-Pearson test for change in log return model
    stat_change_logr, p_change_logr \
        = sc.stats.normaltest(df['change_logr'], nan_policy='omit')
    print(f'p_change_logr of {ticker} is', p_change_logr)


    if p_logr <= alpha:

        # Testing whether normal skewness
        stat_skew, p_skew = sc.stats.skewtest(df['logr'], nan_policy='omit')
        print(f'p_skew of {ticker} is', p_skew)

        # Testing whether normal kurtosis.
        stat_kur, p_kur = sc.stats.kurtosistest(df['logr'],
                                                nan_policy='omit')
        print(f'p_kur of {ticker} is', p_kur)

        if p_skew <= alpha and p_kur <= alpha:

            loc = np.nanmean(df['logr'])
            scale = np.nanstd(df['logr'])

            numpy.random.seed(seed)
            logr = np.random.normal(loc, scale, size=(simulation,pdays))

        elif p_skew <= alpha < p_kur:

            mean = np.nanmean(df['logr'])
            std = np.nanstd(df['logr'])

            deg_free = df['logr'].count() - 1
            loc = mean
            scale = (std**2*(deg_free-2)/deg_free)**0.5

            logr = sc.stats.t.rvs(deg_free, loc, scale,
                                  size=(simulation,pdays),
                                  random_state=seed)

        elif p_skew > alpha >= p_kur:

            mean = np.nanmean(df['logr'])
            std = np.nanstd(df['logr'])
            skew = sc.stats.skew(df['logr'], nan_policy='omit')

            theta = (np.pi/2 * abs(skew)**(2/3)
                     / (abs(skew)**(2/3) + ((4-np.pi)/2)**(2/3)))**0.5 \
                    * np.sign(skew)
            a = theta / (1-theta**2)**0.5
            scale = (std**2/(1-2*theta**2/np.pi))**0.5
            loc = mean - scale*theta*(2/np.pi)**0.5

            logr = sc.stats.skewnorm.rvs(a, loc, scale,
                                         size=(simulation,pdays),
                                         random_state=seed)

        else:
            mean = np.nanmean(df['logr'])
            std = np.nanstd(df['logr'])
            deg_free = df['logr'].count() - 1

            loc = mean
            scale = std
            nc = mean*(1-3/(4*deg_free-1))

            logr = sc.stats.nct.rvs(deg_free, nc, loc, scale,
                                    size=(simulation,pdays),
                                    random_state=seed)

        # Convert logr back to simulated price
        price_t = df['close'].loc[
            df['trading_date'] == df['trading_date'].max()].iloc[0]
        simulated_price = np.zeros(shape=(simulation,pdays), dtype=np.int64)
        for i in range(simulation):
            simulated_price[i, 0] = np.exp(logr[i, 0]) * price_t
            for j in range(1, pdays):
                simulated_price[i, j] \
                    = np.exp(logr[i, j]) * simulated_price[i, j-1]
        df_historical \
            = df[['trading_date', 'close']].iloc[
              int(max(-254,-df['trading_date'].count())):]


    elif p_change_logr <= alpha:

        # Testing whether normal skewness
        stat_skew, p_skew \
            = sc.stats.skewtest(df['change_logr'], nan_policy='omit')
        print(f'p_skew of {ticker} is', p_skew)

        # Testing whether normal kurtosis
        stat_kur, p_kur \
            = sc.stats.kurtosistest(df['change_logr'], nan_policy='omit')
        print(f'p_kur of {ticker} is', p_kur)

        if p_skew <= alpha and p_kur <= alpha:

            loc = 0
            scale = np.nanstd(df['change_logr'])

            numpy.random.seed(seed)
            change_logr \
                = np.random.normal(loc, scale, size=(simulation,pdays))

        elif p_skew <= alpha < p_kur:

            mean = 0
            std = np.nanstd(df['change_logr'])

            deg_free = df['change_logr'].count() - 1
            loc = mean
            scale = (std**2*(deg_free-2)/deg_free)**0.5

            change_logr = sc.stats.t.rvs(deg_free, loc, scale,
                                         size=(simulation,pdays),
                                         random_state=seed)

        elif p_skew > alpha >= p_kur:

            mean = 0
            std = np.nanstd(df['change_logr'])
            skew = sc.stats.skew(df['change_logr'], nan_policy='omit')

            theta = (np.pi/2 * skew**(2/3)
                     / (skew**(2/3) + ((4-np.pi)/2)**(2/3)))**0.5 \
                    * np.sign(skew)
            a = theta / (1-theta**2)**0.5
            scale = (std**2/(1-2*theta**2/np.pi))**0.5
            loc = mean - scale*theta*(2/np.pi)**0.5

            change_logr = sc.stats.skewnorm.rvs(a, loc, scale,
                                                size=(simulation,pdays),
                                                random_state=seed)

        else:
            mean = 0
            std = np.nanstd(df['change_logr'])
            deg_free = df['change_logr'].count() - 1

            loc = mean
            scale = std
            nc = mean*(1-3/(4*deg_free-1))

            change_logr = sc.stats.nct.rvs(deg_free, nc, loc, scale,
                                           size=(simulation,pdays),
                                           random_state=seed)

        # Convert change_logr back to simulated price
        price_t \
            = df['close'].loc[df['trading_date']
                              == df['trading_date'].max()].iloc[0]

        simulated_price = np.zeros(shape=(simulation,pdays),
                                   dtype=np.int64)
        for i in range(simulation):
            simulated_price[i, 0] \
                = np.exp(change_logr[i, 0]) * price_t
            for j in range(1, pdays):
                simulated_price[i, j] \
                    = np.exp(change_logr[i, j]) * simulated_price[i, j-1]
        df_historical \
            = df[['trading_date', 'close']].iloc[
              int(max(-254,-df['trading_date'].count())):]

    else:
        print(f'p_logr of {ticker} is', p_logr)
        print(f'p_change_logr of {ticker} is', p_change_logr)
        raise KeyError(f'{ticker} cannot be simulated'
                       f' with given significance level')

    # Post-processing and graphing
    pro_days = list()
    for j in range(pdays * 2):
        if pd.to_datetime(df['trading_date'].max()
                          + pd.Timedelta(days=j + 1)).weekday() < 5:
            pro_days.append(df['trading_date'].max()
                            + pd.Timedelta(days=j + 1))
    pro_days = pro_days[:pdays]

    simulation_no = [i for i in range(1, simulation + 1)]
    df_simulated_price \
        = pd.DataFrame(data=simulated_price,
                       columns=pro_days,
                       index=simulation_no).transpose()
    last_price = df_simulated_price.iloc[-1,:]
    ubound = pd.DataFrame(df_simulated_price
                          .quantile(q=q_upper, axis=1,
                                    interpolation='linear'))
    dbound = pd.DataFrame(df_simulated_price
                          .quantile(q=q_lower, axis=1,
                                    interpolation='linear'))
    breakeven_price = dbound.min().iloc[0]
    connect_date = pd.date_range(df['trading_date'].max(),
                                 pro_days[0])[[0,-1]]
    connect = pd.DataFrame({'price_u': [price_t, ubound.iloc[0,0]],
                            'price_d': [price_t, dbound.iloc[0,0]]},
                           index=connect_date)
    graph_ticker()

    print(f'Breakeven price of {ticker} is ' + str(breakeven_price))
    print("The execution time is: %s seconds" %(int(time.time()-start_time)))

    if fulloutput is False:
        return breakeven_price
    else:
        input_ = dict()
        input_['breakeven_price'] = breakeven_price   # float
        input_['historical_price'] \
            = pd.DataFrame(df_historical['close'].to_numpy(),
                           index=df_historical['trading_date'].to_numpy())
        input_['simulated_price'] = df_simulated_price.T
        input_['last_price'] = last_price.to_numpy()
        input_['ubound'] = ubound   # pd.DataFrame
        input_['dbound'] = dbound   # pd.DataFrame
        return input_
