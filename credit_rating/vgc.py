from request_phs import *
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from os.path import dirname, realpath
import itertools
matplotlib.use('TkAgg')

destination_dir = join(dirname(realpath(__file__)), 'result')

# Parameters:
centroids = 3
min_tickers = 3
nstd_bound = 2

start_time = time.time()

np.set_printoptions(linewidth=np.inf, precision=4, suppress=True,
                    threshold=sys.maxsize)
pd.set_option("display.max_rows", sys.maxsize,
              "display.max_columns", sys.maxsize,
              'display.expand_frame_repr', True)
pd.options.mode.chained_assignment = None

agg_data = fa.all('gen')  #################### expensive

quantities = ['revenue', 'cogs', 'gross_profit', 'interest',
              'pbt', 'net_income', 'cur_asset', 'cash', 'ar', 'inv',
              'ppe', 'asset', 'liability', 'cur_liability', 'lt_debt',
              'equity']
periods = fa.periods
tickers = fa.tickers('gen')
standards = fa.standards

years = list()
quarters = list()
for period in periods:
    years.append(int(period[:4]))
    quarters.append(int(period[-1]))

period_tuple = list(zip(years,quarters))
inds = [x for x in itertools.product(period_tuple, tickers)]

for i in range(len(inds)):
    inds[i] = inds[i][0] + tuple([inds[i][1]])

index = pd.MultiIndex.from_tuples(inds, names=['year', 'quarter', 'fs'])
col = pd.Index(quantities, name='quantity')

df = pd.DataFrame(np.zeros((len(index), len(col))), columns=col, index=index)

for year, quarter in period_tuple:
    for ticker in tickers:
        for quantity in quantities:
            try:
                if quantity == 'revenue':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('is', '3.')]
                if quantity == 'cogs':
                    df.loc[(year, quarter, ticker), quantity] \
                        = -agg_data.loc[(year, quarter, ticker),
                                        ('is', '4.')]
                if quantity == 'gross_profit':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('is', '5.')]
                if quantity == 'interest':
                    df.loc[(year, quarter, ticker), quantity] \
                        = -agg_data.loc[(year, quarter, ticker),
                                        ('is', '7.1.')]
                if quantity == 'pbt':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('is', '16.')]
                if quantity == 'net_income':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('is','18.')]
                if quantity == 'cur_asset':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'A.I.')]
                if quantity == 'cash':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'A.I.1.')]
                if quantity == 'ar':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'A.I.3.')]
                if quantity == 'inv':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'A.I.4.')]
                if quantity == 'ppe':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'A.II.2.')]
                if quantity == 'asset':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'A.')]
                if quantity == 'liability':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'B.I.')]
                if quantity == 'cur_liability':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'B.I.1.')]
                if quantity == 'lt_debt':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'B.I.2.8.')]
                if quantity == 'equity':
                    df.loc[(year, quarter, ticker), quantity] \
                        = agg_data.loc[(year, quarter, ticker),
                                       ('bs', 'B.II.')]
                else:
                    pass
            except KeyError:
                continue

#del agg_data # for memory savings

df['cur_ratio'] = df['cur_asset'] / df['cur_liability']
df['quick_ratio'] = (df['cur_asset'] - df['inv']) / df['cur_liability']
df['cash_ratio'] = df['cash'] / df['cur_liability']
df['wc_turnover'] = df['revenue'] / (df['cur_asset'] - df['cur_liability'])
df['inv_turnover'] = df['cogs'] / df['inv']
df['ar_turnover'] = df['revenue'] / df['ar']
df['ppe_turnover'] = df['revenue'] / df['ppe']
df['(-) lib/asset'] = -df['liability'] / df['asset']
df['(-) lt_debt/equity'] = -df['lt_debt'] / df['equity']
df['gross_margin'] = df['gross_profit'] / df['revenue']
df['net_margin'] = df['net_income'] / df['revenue']
df['roe'] = df['net_income'] / df['equity']
df['roa'] = df['net_income'] / df['asset']
df['ebit/int'] = (df['pbt'] + df['interest']) / df['interest']

df = df.drop(columns=quantities)
df.sort_index(axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

quantities_new = [i for i in df.columns]

for year, quarter in zip(years, quarters):
    for ticker in tickers:
        for quantity in quantities_new:
            if df.loc[(year, quarter, ticker), quantity] \
                    in [np.nan, np.inf, -np.inf] \
                    or pd.isna(df.loc[(year, quarter, ticker), quantity]):
                df.loc[(year, quarter, ticker), quantity] = np.nan

def compare_industry(ticker:str, standard:str, level:int):
    full_list = fa.classification(standard).iloc[:,level-1]
    industry = full_list.loc[ticker]
    peers = full_list.loc[full_list == industry].index.tolist()

    table = df.loc[df.index.get_level_values(2).isin(peers)]
    table.dropna(axis=0, how='all', inplace=True)

    median = table.groupby(axis=0, level=[0,1]).median()
    quantities = table.xs(ticker, axis=0, level=2)

    comparison = pd.concat([quantities, median], axis=1, join='outer',
                           keys=[ticker, 'median'])

    fig, ax = plt.subplots(3,5, figsize=(18,8),
                           tight_layout=True)

    periods = [str(q[0]) + 'q' + str(q[1]) for q in comparison.index]
    variables \
        = comparison.columns.get_level_values(1).drop_duplicates().to_numpy()
    variables = np.append(variables, None)
    variables = np.reshape(variables, (3,5))
    for row in range(3):
        for col in range(5):
            w = 0.35
            l = np.arange(11)  # the label locations
            if variables[row,col] is None:
                ax[row, col].axis('off')
            else:
                ax[row,col].bar(l-w/2, quantities.iloc[:, row*5+col],
                                width=w, label=ticker,
                                color='tab:orange', edgecolor='black')
                print(quantities.iloc[:, row*4+col])
                ax[row,col].bar(l+w/2, median.iloc[:, row*5+col],
                                width=w, label='Industry\'s Average',
                                color='tab:blue', edgecolor='black')
                plt.setp(ax[row,col].xaxis.get_majorticklabels(), rotation=45)
                ax[row,col].set_xticks(l)
                ax[row,col].set_xticklabels(periods, fontsize=7)
                ax[row, col].set_yticks([])
                ax[row,col].set_autoscaley_on(True)
                ax[row,col].set_title(variables[row,col], fontsize=9)

    fig.suptitle(f'{ticker} \n Comparison with the industry\'s average',
                 fontweight='bold', color='darkslategrey',
                 fontfamily='Times New Roman', fontsize=14)
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left',
               bbox_to_anchor=(0.01, 0.98), ncol=2, fontsize=9,
               markerscale=0.7)

    plt.show()


