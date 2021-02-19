from request_phs import *
import scipy as sc
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as transforms
matplotlib.use('Agg')
plt.ioff()

destination_dir \
    = join(dirname(realpath(__file__)), 'monte_carlo_result')

q_upper = 0.95
q_lower = 0

def monte_carlo(ticker:str, days=66, alpha=0.05,
                simulation=100000, savefigure:bool=True,
                fulloutput:bool=False):

    start_time = time.time()

    global destination_dir

    def reformat_large_tick_values(tick_val, pos):
        """
        Turns large tick values (in the billions, millions and thousands)
        such as 4500 into 4.5K and also appropriately turns 4000 into 4K
        (no zero after the decimal)
        """
        if tick_val >= 1000:
            val = round(tick_val/1000, 1)
            if tick_val%1000 > 0:
                new_tick_format = '{:,}K'.format(val)
            else:
                new_tick_format = '{:,}K'.format(int(val))
        else:
            new_tick_format = int(tick_val)
        new_tick_format = str(new_tick_format)
        return new_tick_format

    def graph_ticker():

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1.plot(df_historical['trading_date'], df_historical['close'],
                 color='darkblue')
        ax1.plot(ubound, color='orange', alpha=5)
        ax1.plot(dbound, color='orange', alpha=5)
        ax1.plot(connect, color='orange', alpha=5)
        ax1.fill_between(pro_days, ubound.iloc[:, 0], dbound.iloc[:, 0],
                         color='green', alpha=0.1)
        ax1.set_title(ticker)
        ax1.set_ylabel('Stock Price')
        ax1.xaxis.set_major_formatter(DateFormatter('%d/%m/%Y'))
        plt.xticks(rotation=15)
        ax1.axhline(breakeven_price, ls='--', linewidth=0.5, color='red')
        ax1.annotate('Breakeven Price: ' + adjprice(breakeven_price),
                     xy=(0.8,breakeven_price),
                     xycoords=transforms
                     .blended_transform_factory(ax1.transAxes, ax1.transData),
                     xytext=(0,-2), textcoords='offset pixels',
                     ha='left', va='top', fontsize=7)
        ax1.yaxis\
            .set_major_formatter(matplotlib.ticker\
            .FuncFormatter(
            reformat_large_tick_values))
        ax1.annotate('Worst Case over \n'
                     + f'{int(simulation):,d}' + ' Simulations: '
                     + f'{round((breakeven_price/price_t-1)*100, 2):,} % \n'
                     + 'Trading Days: ' + f'{days} \n'
                     + 'Breakeven Price: ' + adjprice(breakeven_price),
                     xy=(0.7, 1.01),
                     xycoords=ax1.transAxes,
                     ha='left', va='bottom', fontsize=6)
        if savefigure is True:
            plt.savefig(join(destination_dir,f'{ticker}_result_1.png'),
                        bbox_inches='tight')

        fig2, ax2 = plt.subplots(1, 2, figsize=(8, 5))
        fig2.suptitle('Projected Stock Price: ' + ticker)
        fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.15,
                             top=0.9, wspace=0.15)

        sns.histplot(last_price, ax=ax2[0], bins=100,
                     legend=False, color='darkblue', stat='density')
        ax2[0].set_xlabel('Stock Price')
        ax2[0].set_ylabel('Density')
        ax2[0].axvline(breakeven_price, ls='--', linewidth=0.5,
                       color='red')

        ax2[0].annotate('Breakeven Price:\n' + adjprice(breakeven_price),
                        xy=(breakeven_price,0.85),
                        xycoords=transforms.blended_transform_factory\
                            (ax2[0].transData,ax2[0].transAxes),
                        xytext=(2,0), textcoords='offset pixels',
                        ha='left', va='center', fontsize=8)

        ax2[0].tick_params(axis='y', left=False, labelleft=False)
        ax2[0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))

        sns.ecdfplot(last_price, stat='proportion', ax=ax2[1],
                     legend=False, color='black')
        ax2[1].set_xlabel('Stock Price')
        ax2[1].set_ylabel('Probability')
        # ax2[1].axhline(0.01, ls='--', linewidth=0.5, color='red')
        ax2[1].axvline(breakeven_price, ls='--', linewidth=0.5,
                       color='red')

        ax2[1].annotate('Breakeven Price:\n' + adjprice(breakeven_price),
                        xy=(breakeven_price,0.85),
                        xycoords=transforms
                        .blended_transform_factory\
                            (ax2[1].transData,ax2[1].transAxes),
                        xytext=(2,0), textcoords='offset pixels',
                        ha='left', va='center', fontsize=8)

        ax2[1].tick_params(axis='y', left=False, labelleft=False)
        ax2[1].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))
        if savefigure is True:
            plt.savefig(join(destination_dir,f'{ticker}_result_2.png'),
                             bbox_inches='tight')
        return

    # customize display for numpy and pandas
    np.set_printoptions(linewidth=np.inf, precision=0,
                        suppress=True, threshold=int(10e10))
    pd.set_option("display.max_rows", sys.maxsize,
                  "display.max_column", sys.maxsize,
                  'display.width', None,
                  'display.max_colwidth', 20)

    df = ta.hist(ticker)

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
    stat_change_logr, p_change_logr\
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
            logr = np.random.normal(loc, scale, size=(simulation, days))

        elif p_skew <= alpha < p_kur:

            mean = np.nanmean(df['logr'])
            std = np.nanstd(df['logr'])

            deg_free = df['logr'].count() - 1
            loc = mean
            scale = (std**2*(deg_free-2)/deg_free)**0.5

            logr = sc.stats.t.rvs(deg_free, loc, scale,
                                   size=(simulation, days))

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
                                          size=(simulation, days))

        else:
            mean = np.nanmean(df['logr'])
            std = np.nanstd(df['logr'])
            deg_free = df['logr'].count() - 1

            loc = mean
            scale = std
            nc = mean*(1-3/(4*deg_free-1))

            logr = sc.stats.nct.rvs(deg_free, nc, loc, scale,
                                     size=(simulation, days))

        # Convert logr back to simulated price
        price_t = df['close'].loc[
            df['trading_date'] == df['trading_date'].max()].iloc[0]
        simulated_price = np.zeros(shape=(simulation, days), dtype=np.int64)
        for i in range(simulation):
            simulated_price[i, 0] = np.exp(logr[i, 0]) * price_t
            for j in range(1, days):
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
            change_logr \
                = np.random.normal(loc, scale, size=(simulation, days))

        elif p_skew <= alpha < p_kur:

            mean = 0
            std = np.nanstd(df['change_logr'])

            deg_free = df['change_logr'].count() - 1
            loc = mean
            scale = (std**2*(deg_free-2)/deg_free)**0.5

            change_logr = sc.stats.t.rvs(deg_free, loc, scale,
                                         size=(simulation, days))

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
                                                size=(simulation, days))

        else:
            mean = 0
            std = np.nanstd(df['change_logr'])
            deg_free = df['change_logr'].count() - 1

            loc = mean
            scale = std
            nc = mean*(1-3/(4*deg_free-1))

            change_logr = sc.stats.nct.rvs(deg_free, nc, loc, scale,
                                           size=(simulation, days))

        # Convert change_logr back to simulated price
        price_t \
            = df['close'].loc[df['trading_date']
                              == df['trading_date'].max()].iloc[0]

        simulated_price = np.zeros(shape=(simulation, days),
                                   dtype=np.int64)
        for i in range(simulation):
            simulated_price[i, 0] \
                = np.exp(change_logr[i, 0]) * price_t
            for j in range(1, days):
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
    for j in range(days * 2):
        if pd.to_datetime(df['trading_date'].max()
                          + pd.Timedelta(days=j + 1)).weekday() < 5:
            pro_days.append(df['trading_date'].max()
                            + pd.Timedelta(days=j + 1))
    pro_days = pro_days[:days]

    simulation_no = [i for i in range(1, simulation + 1)]
    df_simulated_price \
        = pd.DataFrame(data=simulated_price,
                       columns=pro_days,
                       index=simulation_no).transpose()
    last_price = df_simulated_price.iloc[-1, :]
    ubound = pd.DataFrame(df_simulated_price
                          .quantile(q=q_upper, axis=1,
                                    interpolation='linear'))
    dbound = pd.DataFrame(df_simulated_price
                          .quantile(q=q_lower, axis=1,
                                    interpolation='linear'))
    breakeven_price = dbound.min().iloc[0]
    connect_date = pd.date_range(df['trading_date'].max(),
                                 pro_days[0])[[0,-1]]
    connect = pd.DataFrame({'price_u': [price_t, ubound.iloc[0, 0]],
                            'price_d': [price_t, dbound.iloc[0, 0]]},
                           index=connect_date)
    graph_ticker()
    plt.close('all')

    print(f'Breakeven price of {ticker} is ' + str(breakeven_price))
    print("The execution time is: %s seconds" %(time.time()-start_time))

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
