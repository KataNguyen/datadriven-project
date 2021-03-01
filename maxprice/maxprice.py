from request_phs import *
from breakeven_price.monte_carlo import *

def maxprice(ticker:str, standard:str, level:int, savefigure:bool=True):

    destination_dir = join(dirname(realpath(__file__)), 'result')

    # from breakeven price
    input_ = monte_carlo(ticker, savefigure=False, fulloutput=True)
    breakeven_price = input_['breakeven_price']
    historical_price = input_['historical_price']
    last_price = input_['last_price']

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
    q_upper = 0.95
    q_lower = 0
    upper = np.quantile(last_price, q=q_upper)
    lower = np.quantile(last_price, q=q_lower)
    last_price_truncated = last_price
    last_price_truncated = last_price_truncated[last_price_truncated >= lower]
    last_price_truncated = last_price_truncated[last_price_truncated <= upper]

    price_D = np.quantile(last_price_truncated, q=0.01)
    price_C = np.quantile(last_price_truncated, q=0.15)
    price_B = np.quantile(last_price_truncated, q=0.55)
    price_A = np.quantile(last_price_truncated, q=1)

    if 0 <= score <= 25:
        prange = last_price_truncated
        prange = prange[prange >= breakeven_price]
        prange = prange[prange <= price_D]
        maxprice = np.percentile(prange, q=(score-0)/25*100)
        rating = 'D'
    elif 25 <= score <= 50:
        prange = last_price_truncated
        prange = prange[prange >= price_D]
        prange = prange[prange <= price_C]
        maxprice = np.percentile(prange, q=(score-25)/25*100)
        rating = 'C'
    elif 50 <= score <= 75:
        prange = last_price_truncated
        prange = prange[prange >= price_C]
        prange = prange[prange <= price_B]
        maxprice = np.percentile(prange, q=(score-50)/25*100)
        rating = 'B'
    elif 75 <= score <= 100:
        prange = last_price_truncated
        prange = prange[prange >= price_B]
        prange = prange[prange <= price_A]
        maxprice = np.percentile(prange, q=(score-75)/25*100)
        rating = 'A'
    else:
        maxprice = None
        rating = 'Unclassifiable' # might never happen

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

        Acolor = 'green'
        Bcolor = 'olivedrab'
        Ccolor = 'darkorange'
        Dcolor = 'firebrick'

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Max Price Analysis: ' + ticker, ha='center',
                     fontweight='bold', color='darkslategrey',
                     fontfamily='Arial', fontsize=17)
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15,
                             top=0.9, wspace=0.15)

        sns.histplot(last_price, ax=ax[0], bins=100,
                     legend=False, color='tab:blue', stat='density')
        sns.kdeplot(last_price, ax=ax[0])

        y_upper = ax[0].get_ylim()[1]*1.03
        y_lower = ax[0].get_ylim()[0]
        ax[0].set_ylim(y_lower, y_upper)
        ax[0].fill_betweenx([y_lower, y_upper], breakeven_price, price_D,
                            color=Dcolor, alpha=0.2)
        ax[0].fill_betweenx([y_lower, y_upper], price_D, price_C,
                            color=Ccolor, alpha=0.2)
        ax[0].fill_betweenx([y_lower, y_upper], price_C, price_B,
                            color=Bcolor, alpha=0.2)
        ax[0].fill_betweenx([y_lower, y_upper], price_B, price_A,
                            color=Acolor, alpha=0.2)

        ax[0].set_xlabel('Stock Price')
        ax[0].set_ylabel('Density')
        ax[0].axvline(breakeven_price, ls='-', linewidth=2,
                       color=Dcolor)
        ax[0].annotate('Breakeven Price:\n' + f'{adjprice(breakeven_price)}',
                       xy=(breakeven_price, 0.9),
                       xycoords=transforms.blended_transform_factory(
                           ax[0].transData, ax[0].transAxes),
                       xytext=(3,0), # 3 pixels horizontal offset
                       textcoords='offset pixels',
                       ha='left', fontsize=7)

        if score <= 25:
            mpcolor = Dcolor
        elif score <= 50:
            mpcolor = Ccolor
        elif score <= 75:
            mpcolor = Bcolor
        else:
            mpcolor = Acolor

        ax[0].axvline(maxprice, linestyle='-', linewidth=2,
                      color=mpcolor)
        ax[0].annotate('Max Price:\n' + f'{adjprice(maxprice)}',
                       xy=(maxprice, 0.9),
                       xycoords=transforms.blended_transform_factory(
                           ax[0].transData, ax[0].transAxes),
                       xytext=(3,0), # 3 pixels horizontal offset
                       textcoords='offset pixels',
                       ha='left', fontsize=7)

        ax[0].annotate('D',
                       xy=(np.mean([breakeven_price, price_D]), 1.01),
                       xycoords=transforms.blended_transform_factory(
                           ax[0].transData, ax[0].transAxes),
                       ha='center', color=Dcolor, fontsize=10)
        ax[0].annotate('C',
                       xy=(np.mean([price_D, price_C]), 1.01),
                       xycoords=transforms.blended_transform_factory(
                           ax[0].transData, ax[0].transAxes),
                       ha='center', color=Ccolor, fontsize=10)

        ax[0].annotate('B',
                       xy=(np.mean([price_C, price_B]), 1.01),
                       xycoords=transforms.blended_transform_factory(
                           ax[0].transData, ax[0].transAxes),
                       ha='center', color=Bcolor, fontsize=10)

        ax[0].annotate('A',
                       xy=(np.mean([price_B, price_A]), 1.01),
                       xycoords=transforms.blended_transform_factory(
                           ax[0].transData, ax[0].transAxes),
                       ha='center', color=Acolor, fontsize=10)

        ref_price = historical_price.iloc[-1,-1]
        if maxprice < ref_price: dorp = '% Discount: '
        else: dorp = '% Premium: '
        ax[0].annotate(f'Reference Price: '
                       + f'{adjprice(ref_price)} \n'
                       + f'{int((q_upper-q_lower)*100)}% Confidence Interval:\n'
                       + f'{adjprice(breakeven_price)}' + ' - '
                       + f'{adjprice(price_A)} \n'
                       + '------------------------- \n'
                       + f'Credit Score: {score:.0f} \n'
                       + f'Credit Rating: {rating} \n'
                       + '------------------------- \n'
                       + f'Breakeven Price: {adjprice(breakeven_price)} \n'
                       + f'Max Price: {adjprice(maxprice)} \n'
                       + dorp + f'{(maxprice/ref_price-1)*100:.2f}% \n'
                       + f'Mark-up Value: {adjprice(maxprice-breakeven_price)} \n'
                       + f'% Mark-up: {(maxprice/breakeven_price-1)*100:.2f}%',
                       xy=(0.72, 0.95),
                       xycoords=ax[0].transAxes,
                       ha='left', va='top', color='black', fontsize=7)

        ax[0].tick_params(axis='y', left=False, labelleft=False)
        ax[0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))

        #===================================================

        sns.ecdfplot(last_price, stat='proportion', ax=ax[1],
                     legend=False, color='black')

        ax[1].set_xlabel('Stock Price')
        ax[1].set_ylabel('Probability')

        adjust_multiplier = 1.01
        yrange = ax[1].get_ylim()
        y_upper = yrange[1] * adjust_multiplier
        y_lower = yrange[0]
        ax[1].set_ylim(y_lower, y_upper)

        ax[1].axvline(breakeven_price, ls='-', linewidth=2,
                       color=Dcolor)

        ax[1].axvline(maxprice, linestyle='-', linewidth=2,
                      color=mpcolor)

        def f(start:float, stop:float, array:np.array, num:int=100):
            x_val = np.linspace(start, stop, num)
            y_val = np.array([np.sum(array<=x) for x in x_val]) \
                             / len(array)
            return x_val, y_val

        def nearest(value:float, array:np.array):
            distance = np.abs(array-value)
            index = distance.argmin()
            nearest_point = array[index]
            return nearest_point

        xy = f(breakeven_price, maxprice, last_price)
        ax[1].fill_between(xy[0], 0, xy[1],
                           color='tab:red', alpha=0.2)

        downside_risk = xy[1][-1]
        y_loc = downside_risk/4
        x_loc = (xy[0][xy[1] == nearest(y_loc, xy[1])][0] +  maxprice)/2
        ax[1].annotate(f'{round(downside_risk*100)}%',
                       xy=(x_loc, y_loc),
                       xycoords= ax[1].transData,
                       ha='center', va='center', fontsize=10, color='tab:red',
                       fontweight= 'bold')

        xy = f(maxprice, last_price.max(), last_price)
        ax[1].fill_between(xy[0], xy[1], 1*adjust_multiplier,
                           color='tab:green', alpha=0.2)

        upside_risk = 1 - downside_risk
        y_loc = downside_risk + upside_risk*4/5
        x_loc = (xy[0][xy[1] == nearest(y_loc, xy[1])][0] +  maxprice)/2
        ax[1].annotate(f'{round(upside_risk*100)}%',
                       xy=(x_loc, y_loc),
                       xycoords= ax[1].transData,
                       ha='center', va='center', fontsize=10, color='g',
                       fontweight= 'bold')

        ax[1].annotate(f'Downside Risk: {round(downside_risk*100)}% \n'
                       + 'This risk measure the likelihood \n'
                       + 'of maxprice\'s deterioration \n'
                       + 'in response to altering credit score \n'
                       + '-------------------------\n'
                       + f'Upside Risk: {round(upside_risk*100)}% \n'
                       + 'This risk measure the likelihood \n'
                       + 'of maxprice\'s enhancement \n'
                       + 'in response to altering credit score',
                       xy=(0.62, 0.4),
                       xycoords=ax[1].transAxes,
                       ha='left', va='center', color='black', fontsize=7)

        ax[1].tick_params(axis='y', left=False, labelleft=False)
        ax[1].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(reformat_large_tick_values))

        if savefigure is True:
            plt.savefig(join(destination_dir, f'{ticker}_maxprice.png'),
                        bbox_inches='tight')

    graph_maxprice()

    return maxprice
