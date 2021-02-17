from request_phs import *
from breakeven_price.monte_carlo import *


def maxprice(ticker:str, standard:str, level:int, savefigure:bool=True):

    destination_dir = join(realpath(__file__), 'result')

    # from breakeven price
    input_ = monte_carlo(ticker, savefigure=False, fulloutput=True)
    breakeven_price = input_['breakeven_price']
    historical_price = input_['historical_price']
    simulated_price = input_['simulated_price']
    last_price = input_['last_price']
    ubound = input_['ubound']
    dbound = input_['dbound']

    # from credit rating
    credit_file = join(dirname(dirname(realpath(__file__))),
                       'credit_rating', 'result', 'result_table.csv')
    credit_result = pd.read_csv(credit_file, index_col='ticker')
    credit_result = credit_result.loc[ticker]
    credit_result \
        = credit_result.loc[credit_result['standard'] == standard]
    credit_result \
        = credit_result.loc[
        credit_result['level'] == standard + '_l' + str(level)]
    score = credit_result.iloc[0,-1]

    # combine to produce maxprice
    upper = np.quantile(last_price, q=0.99)
    lower = np.quantile(last_price, q=0)
    last_price = last_price[last_price >= lower]
    last_price = last_price[last_price <= upper]
    maxprice = np.percentile(last_price, q=score)



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

    def graph_maxprice():

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle('Projected Stock Price: ' + ticker)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15,
                             top=0.9, wspace=0.15)

        sns.histplot(simulated_price, ax=ax[0], bins=100,
                     legend=False, color='tab:blue', stat='density')
        ax[0].set_xlabel('Stock Price')
        ax[0].set_ylabel('Density')
        ax[0].axvline(breakeven_price, ls='--', linewidth=0.5,
                       color='red')
        ax[0].text(breakeven_price*1.05, 0.85,
                    'Breakeven Price:\n'+str(
                        f"{round(breakeven_price):,d}"), fontsize=8,
                    transform=transforms.blended_transform_factory(
                        ax[0].transData, ax[0].transAxes))
        ax[0].tick_params(axis='y', left=False, labelleft=False)
        ax[0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))

        sns.ecdfplot(simulated_price, stat='proportion', ax=ax[1],
                     legend=False, color='black')
        ax[1].set_xlabel('Stock Price')
        ax[1].set_ylabel('Probability')
        # ax[1].axhline(0.01, ls='--', linewidth=0.5, color='red')
        ax[1].axvline(breakeven_price, ls='--', linewidth=0.5,
                       color='red')
        ax[1].text(breakeven_price*1.05, 0.85,
                   'Breakeven Price:\n'+str(
                       f"{round(breakeven_price):,d}"), fontsize=8,
                   transform=transforms.blended_transform_factory(
                       ax[0].transData, ax[0].transAxes))
        ax[1].tick_params(axis='y', left=False, labelleft=False)
        ax[1].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))
        if savefigure is True:
            plt.savefig(join(destination_dir, f'{ticker}_maxprice.png'),
                        bbox_inches='tight')

    graph_maxprice()

    return maxprice
