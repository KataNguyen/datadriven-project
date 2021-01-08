from request_data import *
import scipy as sc
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.transforms as transforms

def monte_carlo(ticker, days=66, alpha=0.001,
                simulation=100000, graph='on',
                location=join(os.getcwd(),'result')):

    start_time = time.time()

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
        ax1.text(df['trading_date']
                 .iloc[int(max(-200, -df['trading_date'].count()))],
                 breakeven_price*1.02,
                 'Breakeven Price: '
                 +str(f"{round(breakeven_price):,d}"),
                 fontsize=7)
        ax1.yaxis\
            .set_major_formatter(matplotlib.ticker\
            .FuncFormatter(
            reformat_large_tick_values))

        ax1.text(0.7, 1.01, "Worst case: "
                 +'{:,}'.format(round((breakeven_price/price_t-1)*100, 2))
                 +'%', fontsize=6, transform=ax1.transAxes)
        ax1.text(0.7, 1.04, "Working days: "
                 +'{}'.format(days), fontsize=6, transform=ax1.transAxes)
        ax1.text(0.7, 1.07, "Breakeven Price: "
                 +'{:,}'.format(int(breakeven_price)),
                 fontsize=6, transform=ax1.transAxes)
        plt.savefig(location+f'{ticker}_result1.png', bbox_inches='tight')

        fig2, ax2 = plt.subplots(1, 2, figsize=(8, 5))
        fig2.suptitle('Projected Stock Price: '+ticker)
        fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.15,
                             top=0.9, wspace=0.15)

        sns.histplot(price_last, ax=ax2[0], bins=100,
                     legend=False, color='darkblue', stat='density')
        ax2[0].set_xlabel('Stock Price')
        ax2[0].set_ylabel('Density')
        ax2[0].axvline(breakeven_price, ls='--', linewidth=0.5,
                       color='red')
        ax2[0].text(breakeven_price*1.05, 0.85,
                    'Breakeven Price:\n'+str(
                        f"{round(breakeven_price):,d}"), fontsize=8,
                    transform=transforms.blended_transform_factory(
                        ax2[0].transData, ax2[0].transAxes))
        ax2[0].tick_params(axis='y', left=False, labelleft=False)
        ax2[0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))

        sns.ecdfplot(price_last, stat='proportion', ax=ax2[1],
                     legend=False, color='black')
        ax2[1].set_xlabel('Stock Price')
        ax2[1].set_ylabel('Probability')
        # ax2[1].axhline(0.01, ls='--', linewidth=0.5, color='red')
        ax2[1].axvline(breakeven_price, ls='--', linewidth=0.5,
                       color='red')
        ax2[1].text(breakeven_price*1.05, 0.85,
                    'Breakeven Price:\n'+str(
                        f"{round(breakeven_price):,d}"), fontsize=8,
                    transform=transforms.blended_transform_factory(
                        ax2[0].transData, ax2[0].transAxes))
        ax2[1].tick_params(axis='y', left=False, labelleft=False)
        ax2[1].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))
        plt.savefig(location+f'{ticker}_result2.png', bbox_inches='tight')
        return

    # customize display for numpy and pandas
    np.set_printoptions(linewidth=np.inf, precision=0,
                        suppress=True, threshold=int(10e10))
    pd.set_option("display.max_rows", sys.maxsize,
                  "display.max_column", sys.maxsize,
                  'display.width', None,
                  'display.max_colwidth', 20)

    df = request_trading_hist(ticker)

    # cleaning data
    df['trading_date'] \
        = pd.to_datetime(df['trading_date'])
    df['close'].loc[df['close'] == 0] \
        = df['ref'].loc[df['close'] == 0]
    df['ref'].loc[df['ref'] == 0] \
        = df['close'].loc[df['ref'] == 0]
    df['change_percent'] \
        = df['close'] / df['ref'] - 1
    df['log_r'] \
        = np.log(1 + df['change_percent'])
    for i in range(1, len(df.index)):
        df['change_log_r'] \
            = df.loc[:, 'log_r'].iloc[i] - df.loc[:, 'log_r'].iloc[i-1]
    df['change_log_r'].iloc[0] = 0
    df['close'] = df['close'] * 1000
    df.drop(columns=['ref'])

    # Fundamental Statistics:
    mean_logr = np.average(df['log_r'])
    std_logr = np.std(df['log_r'])
    kur_logr = sc.stats.kurtosis(df['log_r'])
    skew_logr = sc.stats.skew(df['log_r'])

    # D'Agostino-Pearson test for log return model
    stat2_logr, p2_logr = sc.stats.normaltest(df['log_r'], nan_policy='omit')
    print(f'p2_logr of {ticker} is', p2_logr)

    if p2_logr <= alpha:

        # Testing whether normal skewness
        stat3_logr, p3_logr = sc.stats.skewtest(df['log_r'], nan_policy='omit')
        print(f'p3_logr of {ticker} is', p3_logr)

        # Testing whether normal kurtosis.
        stat4_logr, p4_logr = sc.stats.kurtosistest(df['log_r'],
                                                    nan_policy='omit')
        print(f'p4_logr of {ticker} is', p4_logr)

        if p3_logr <= alpha and p4_logr <= alpha:
            log_r = np.random.normal(mean_logr, std_logr,
                                     size=(simulation, days))
        elif p3_logr <= alpha and alpha <= p4_logr:
            log_r = sc.stats.t.rvs(df=df['log_r'].count() - 1, loc=mean_logr,
                                   scale=std_logr, size=(simulation, days))
        elif p4_logr <= alpha and alpha <= p3_logr:
            log_r = sc.stats.skewnorm.rvs(a=skew_logr, loc=mean_logr,
                                          scale=std_logr,
                                          size=(simulation, days))
        else:
            log_r = sc.stats.nct.rvs(df=df['log_r'].count() - 1,
                                     nc=skew_logr, loc=mean_logr,
                                     scale=std_logr, size=(simulation, days))

        # Convert log_r back to simulated price
        price_t = df['close'].loc[
            df['trading_date'] == df['trading_date'].max()].iloc[0]
        simulated_price = np.zeros(shape=(simulation, days), dtype=np.int64)
        for i in range(simulation):
            simulated_price[i, 0] = np.exp(log_r[i, 0]) * price_t
            for j in range(1, days):
                simulated_price[i, j] \
                    = np.exp(log_r[i, j]) * simulated_price[i, j-1]
        df_historical \
            = df[['trading_date', 'close']].iloc[
              int(max(-254,-df['trading_date'].count())):]

        # Post-processing and graphing
        pro_days = list()
        for j in range(days*2):
            if pd.to_datetime(df['trading_date'].max()
                              + pd.Timedelta(days=j+1)).weekday() < 5:
                pro_days.append(df['trading_date'].max()
                                + pd.Timedelta(days=j+1))
        pro_days = pro_days[:days]

        simulation_no \
            = [i for i in range(1, simulation+1)]
        df_simulated_price \
            = pd.DataFrame(data=simulated_price,
                           columns=pro_days,
                           index=simulation_no).transpose()
        price_last = df_simulated_price.iloc[-1, :]
        ubound = pd.DataFrame(df_simulated_price\
                              .quantile(q=0.95, axis=1,
                                        interpolation='linear'))
        dbound = pd.DataFrame(df_simulated_price\
                              .quantile(q=0.00, axis=1,
                                        interpolation='linear'))
        breakeven_price = dbound.min().iloc[0]
        connect_date \
            = pd.date_range(df['trading_date'].max(), pro_days[0])[[0, -1]]
        connect \
            = pd.DataFrame({'price_u': [price_t, ubound.iloc[0, 0]],
                            'price_d': [price_t, dbound.iloc[0, 0]]},
                           index=connect_date)

        if graph=='on':
            graph_ticker()

    else:

        # Fundamental Statistics
        mean_change_logr = np.average(df['change_log_r'])
        std_change_logr = np.std(df['change_log_r'])
        kur_change_logr = sc.stats.kurtosis(df['change_log_r'])
        skew_change_logr = sc.stats.skew(df['change_log_r'])

        # D'Agostino-Pearson test for change in log return model
        stat2_change_logr, p2_change_logr \
            = sc.stats.normaltest(df['change_log_r'], nan_policy='omit')
        print(f'p2_change_logr of {ticker} is', p2_change_logr)

        if p2_change_logr <= alpha:

            # Testing whether normal skewness
            stat3_change_logr, p3_change_logr \
                = sc.stats.skewtest(df['change_log_r'], nan_policy='omit')
            print(f'p3_change_logr of {ticker} is', p3_change_logr)

            # Testing whether normal kurtosis
            stat4_change_logr, p4_change_logr \
                = sc.stats.kurtosistest(df['change_log_r'], nan_policy='omit')
            print(f'p4_change_logr of {ticker} is', p4_change_logr)

            if p3_change_logr <= alpha and p4_change_logr <= alpha:
                change_logr \
                    = np.random.normal(loc=mean_change_logr,
                                       scale=std_change_logr,
                                       size=(simulation, days))
            elif p3_change_logr <= alpha and alpha <= p4_change_logr:
                change_logr \
                    = sc.stats.t.rvs(df=df['change_log_r'].count() - 1,
                                     loc=mean_change_logr,
                                     scale=std_change_logr,
                                     size=(simulation, days))
            elif p4_change_logr <= alpha and alpha <= p3_change_logr:
                change_logr \
                    = sc.stats.skewnorm.rvs(a=skew_change_logr,
                                            loc=mean_change_logr,
                                            scale=std_change_logr,
                                            size=(simulation, days))
            else:
                change_logr \
                    = sc.stats.nct.rvs(df=df['change_log_r'].count()
                                          - 1, nc=skew_change_logr,
                                       loc=mean_change_logr,
                                       scale=std_change_logr,
                                       size=(simulation, days))

            # Convert change_logr back to simulated price
            price_t \
                = df['close'].loc[df['trading_date']
                                  == df['trading_date'].max()].iloc[0]
            price_t1 \
                = df['close'].loc[df['trading_date']
                                  == df['trading_date'].max()
                                  - pd.Timedelta('1 day')].iloc[0]
            simulated_price = np.zeros(shape=(simulation, days),
                                       dtype=np.int64)
            for i in range(simulation):
                simulated_price[i, 0] \
                    = np.exp(change_logr[i, 0]) * price_t ** 2 / price_t1
                simulated_price[i, 1] \
                    = np.exp(change_logr[i, 1]) * simulated_price[i, 0] ** 2\
                      / price_t
                for j in range(days):
                    simulated_price[i, j] \
                        = np.exp(change_logr[i, j]) * simulated_price[i, j-1] \
                          ** 2 / simulated_price[i, j-2]
            df_historical \
                = df[['trading_date', 'close']].iloc[
                  int(max(-254,df['trading_date'].count())):]

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
            price_last = df_simulated_price.iloc[-1, :]
            ubound = pd.DataFrame(df_simulated_price
                                  .quantile(q=0.95, axis=1,
                                            interpolation='linear'))
            dbound = pd.DataFrame(df_simulated_price
                                  .quantile(q=0.00, axis=1,
                                            interpolation='linear'))
            breakeven_price = dbound.min().iloc[0]
            connect_date = pd.date_range(df['trading_date'].max(), pro_days[0])
            connect = pd.DataFrame({'price_u': [price_t, ubound.iloc[0, 0]],
                                    'price_d': [price_t, dbound.iloc[0, 0]]},
                                   index=connect_date)

            if graph == 'on':
                graph_ticker()

        else:
            raise ValueError(f'{ticker} cannot be simulated'
                             f' with given significance level')

    print("The execution time is: %s seconds" %(time.time()-start_time))
    return breakeven_price


def montecarlo_simulation_list(stocklist, days=66, alpha=0.001,
                               simulation=100000,
                               location=join(os.getcwd(),'result')):
    df_breakeven_price = pd.DataFrame(columns=['ticker', 'breakeven_price'])
    for i in stocklist:
        try:
            df_breakeven_price \
                = df_breakeven_price.append({'ticker': i, 'breakeven_price':
                monte_carlo(i, days, alpha,
                            simulation,
                            location)}, ignore_index=True)
        except ValueError:
            try:
                df_breakeven_price \
                    = df_breakeven_price.append({'ticker': i,
                'breakeven_price':
                    monte_carlo(i, days, 0.05,
                                simulation,
                                location)}, ignore_index=True)
            except ValueError:
                continue

    return df_breakeven_price


#=====================