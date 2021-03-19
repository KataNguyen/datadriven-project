from function_phs import *
from request_phs import *
from breakeven_price.monte_carlo import *

def maxprice(tickers:list, standard:str='bics', level:int=1,
             savefigure:bool=True):

    # pre-prcessing
    input_folder = join(dirname(dirname(realpath(__file__))),
                           'credit_rating', 'result')
    maxprice_dict = dict()
    for ticker in tickers:
        try:
            segment = fa.segment(ticker)
            file_name = f'result_table_{segment}.csv'
            rating_file = join(input_folder, file_name)

            destination_dir = join(dirname(dirname(realpath(__file__))),
                                   'maxprice', 'result')

            # maxPrice calculation
            last_period = fa.periods[-1]
            rating_result = pd.read_csv(rating_file, index_col='ticker')
            if segment == 'gen':
                rating_result \
                    = rating_result.loc[rating_result['level']
                                        == f'{standard}_l{level}'].T
                rating_result.drop(index=['standard', 'level', 'industry'],
                                   inplace=True)
            else:
                rating_result = rating_result.T

            score = rating_result[[ticker]] ; score.columns = ['score']
            highlow = ta.prhighlow(ticker, fquarters=1)
            plow = highlow['low'][['low_price']] * 1000
            plow = plow.shift(periods=-1, axis=0).loc[score.index]
            phigh = highlow['high'][['high_price']] * 1000
            phigh = phigh.shift(periods=-1, axis=0).loc[score.index]
            table = pd.concat([pd.DataFrame(score),plow], axis=1)
            last_score = table['score'].iloc[-1]
            table = table.head(-1)
            table['multiple'] = table['low_price']/table['score']
            min_multiple = table['multiple'].min()

            def fprice(score):
                return min_multiple*score
            maxprice = fprice(last_score)

            eopdate = seopdate(last_period)[1]
            refprice = ta.hist(ticker,eopdate,eopdate)['close'].iloc[0] * 1000
            discount = maxprice/refprice - 1

            def graph_maxprice1():

                nonlocal rating_result
                x = rating_result[[ticker]]
                ylow = plow
                yhigh = phigh

                fig, ax = plt.subplots(1,1,figsize=(8,6))

                for i in range(x.shape[0]):

                    if x.index[i] == last_period:
                        c = 'tab:red'
                        a = 1
                        fw = 'bold'
                    else:
                        c = 'black'
                        a = 0.8
                        fw = 'normal'

                    ax.scatter(x.iloc[i, 0], ylow.iloc[i, 0],
                               label='Next Quarter\'s Lowest Price',
                               color='firebrick', edgecolors='firebrick',
                               marker='^', alpha=a)
                    ax.scatter(x.iloc[i, 0], yhigh.iloc[i, 0],
                               label='Next Quarter\'s Highest Price',
                               color='forestgreen', edgecolors='forestgreen',
                               marker='v', alpha=a)
                    ax.plot([x.iloc[i, 0], x.iloc[i, 0]],
                            [ylow.iloc[i, 0], yhigh.iloc[i, 0]],
                            color='black', alpha=a)
                    ax.annotate(x.index[i], xy=(x.iloc[i,0],ylow.iloc[i,0]),
                                xycoords=ax.transData,
                                xytext=(3,5),
                                textcoords='offset pixels',
                                ha='left', fontsize=7,
                                color=c, alpha=a, fontweight=fw)
                    ax.set_xlabel('Credit Score', labelpad=10)
                    ax.set_ylabel('Stock Price',rotation=90, labelpad=5)
                    ax.xaxis.grid(True, alpha=0.05)
                    ax.yaxis.grid(True, alpha=0.2)

                ax.xaxis.set_major_locator(MaxNLocator(integer=True,
                                                       steps=[1,2,5]))
                ax.yaxis.set_major_formatter(FuncFormatter(priceKformat))
                ax.axhline(maxprice, ls='--', linewidth=0.5, color='tab:red')
                ax.annotate(f'Max Price = {adjprice(maxprice)}',
                            xy=(0.7,maxprice),
                            xycoords=transforms.blended_transform_factory(
                                ax.transAxes, ax.transData),
                            xytext=(0,3),
                            textcoords='offset pixels',
                            ha='left', fontsize=7,
                            color='tab:red', fontweight='bold')

                ax.set_title(f'{ticker}\n Historical Relation between\n'
                             f'Credit Score and Stock Price',
                             fontfamily='Times New Roman', fontsize=13,
                             fontweight='bold', color='black')

                handles, labels = ax.get_legend_handles_labels()
                handles = handles[1:3] ; labels = labels[1:3]
                ax.legend(handles, labels, loc='best',
                          ncol=1, fontsize=8, numpoints=1,
                          framealpha=1)

                if savefigure is True:
                    plt.savefig(join(destination_dir, 'Chart1',
                                     f'{ticker}_chart1.png'),
                                bbox_inches='tight')

            def graph_maxprice2():

                nonlocal rating_result
                peers = fa.peers(ticker, standard, level+1)
                peers = list(set(peers) & set(rating_result.columns))
                # (some ticker were excluded by not having data for CreditRating)
                rating_result = rating_result[peers]

                delta_scores = rating_result.diff(periods=1, axis=0)
                delta_scores = delta_scores.loc[[last_period]].T

                fig, ax = plt.subplots(1,1, figsize=(8,6))

                table = pd.DataFrame(delta_scores)
                table.columns = ['score']
                table[['low_return','high_return']] = np.nan

                folder = 'price' ; file = 'prhighlow.csv'
                highlow = pd.read_csv(join(dirname(dirname(realpath(__file__))),
                                           'database', folder, file),
                                      index_col=[0])
                for ticker_ in peers:
                    if isinstance(highlow.loc[ticker_][-1], str):
                        string = highlow.loc[ticker_][-1]\
                            .replace('(', '').replace(')','')
                        string = string.split(',')
                        table.loc[ticker_,'low_return'] = float(string[3])
                        table.loc[ticker_,'high_return'] = float(string[4])

                        xloc = table.loc[ticker_,'score']
                        ylow = table.loc[ticker_,'low_return']
                        yhigh = table.loc[ticker_,'high_return']

                        if ticker_ == ticker:
                            c = 'tab:red'
                            a = 1
                            fw = 'bold'
                        else:
                            c = 'black'
                            a = 0.2
                            fw = 'normal'

                        ax.scatter(xloc, ylow,
                                   label=f'Lowest Return since eop {last_period}',
                                   color='firebrick', edgecolors='firebrick',
                                   marker='^', alpha=a)
                        ax.scatter(xloc, yhigh,
                                   label=f'Highest Return since eop {last_period}',
                                   color='forestgreen', edgecolors='forestgreen',
                                   marker='v', alpha=a)
                        ax.plot([xloc, xloc], [ylow, yhigh],
                                color='black', alpha=a)
                        ax.annotate(ticker_, xy=(xloc, ylow),
                                    xycoords=ax.transData,
                                    xytext=(0, -20),
                                    textcoords='offset pixels',
                                    ha='center', fontsize=7,
                                    color=c, alpha=a, fontweight=fw)
                    else:
                        pass

                ax.xaxis.set_major_locator(MaxNLocator(min_n_ticks=10,
                                                       integer=True,
                                                       steps=[1,2,5]))
                ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

                ax.axhline(0, ls='--', linewidth=0.5, color='black')
                ax.axvline(0, ls='--', linewidth=0.5, color='black')

                ax.axhline(discount, ls='--',
                           linewidth=0.5, color='tab:red')
                ax.annotate(f'Discount Rate = {discount:.2%}',
                            xy=(0.7,discount),
                            xycoords=transforms.blended_transform_factory(
                                ax.transAxes, ax.transData),
                            xytext=(0, 3),
                            textcoords='offset pixels',
                            ha='left', fontsize=7,
                            color='tab:red', fontweight='bold')

                subtext_cur = '{'+f'{last_period}'+'}'
                subtext_pre = '{'+f'{period_cal(last_period,quarters=-1)}'+'}'
                ax.set_xlabel(fr'$CreditScore_{subtext_cur}  -  $'
                              + fr'$CreditScore_{subtext_pre}$', labelpad=10)
                ax.set_ylabel('Return', labelpad=2, rotation=90)
                ax.xaxis.grid(True, alpha=0.05)
                ax.yaxis.grid(True, alpha=0.2)

                ax.set_title(f'{ticker}\n Cross-sectional Relation between\n'
                             r'$\Delta$Credit Score and Return',
                             fontfamily='Times New Roman', fontsize=13,
                             fontweight='bold', color='black')

                handles, labels = ax.get_legend_handles_labels()
                handles = handles[1:3] ; labels = labels[1:3]
                ax.legend(handles, labels, loc='best',
                          ncol=1, fontsize=8, numpoints=1,
                          framealpha=1)

                if savefigure is True:
                    plt.savefig(join(destination_dir, 'Chart2',
                                     f'{ticker}_chart2.png'),
                                bbox_inches='tight')

                return

            graph_maxprice1()
            graph_maxprice2()
            maxprice_dict[ticker] \
                = int(adjprice(maxprice).replace(',',''))

        except (ValueError, KeyError):
            print(f'{ticker} does not exist in {standard} classification')

    return maxprice_dict
