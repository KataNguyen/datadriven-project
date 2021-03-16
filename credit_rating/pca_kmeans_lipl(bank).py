from phs import *
from request_phs import *

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

agg_data = fa.all('bank')  #################### expensive

quantities = ['balnce_sbv', 'balnce_inst', 'loans_customer', 'trading_sec',
              'invst_sec', 'net_int_income', 'int_income', 'int_expense',
              'net_feecomisn_income', 'ga_expense', 'oper_income',
              'operprofit_bprovisn', 'asset', 'equity', 'liability',
              'deposit_sbv', 'deposit_inst', 'deposit_customer', 'gov_funding',
              'paper_issued', 'substd', 'doubtful', 'baddebt', 'loans',
              'provsn_inst', 'provsn_customer', 'provsn_charge', 'net_income',
              'cash', 'avaisale_sec', 'feecomisn_expense', 'other_expense',
              'feecomisn_income', 'gainlossfxgold', 'gainlosstrading',
              'gainlossinst', 'other_income', 'dividend_income', 'cur_deposit',
              'saving_deposit', 'deposit']

periods = fa.periods
tickers = fa.tickers('bank')

years = list()
quarters = list()
for period in periods:
    years.append(int(period[:4]))
    quarters.append(int(period[-1]))

period_tuple = list(zip(years,quarters))
inds = [x for x in itertools.product(period_tuple, tickers)]

for i in range(len(inds)):
    inds[i] = inds[i][0] + tuple([inds[i][1]])

index = pd.MultiIndex.from_tuples(inds, names=['year', 'quarter', 'ticker'])
col = pd.Index(quantities, name='quantity')

df = pd.DataFrame(columns=col, index=index)

for year, quarter in period_tuple:
    for ticker in tickers:
        for quantity in quantities:
            if quantity == 'balnce_sbv':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.2.')]
            elif quantity == 'balnce_inst':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.3.')]
            elif quantity == 'loans_customer':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.6.')]
            elif quantity == 'trading_sec':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.4.')]
            elif quantity == 'invst_sec':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.8.')]
            elif quantity == 'net_int_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '1.')]
            elif quantity == 'int_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '1.1.')]
            elif quantity == 'int_expense':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '1.2.')]
            elif quantity == 'net_feecomisn_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '2.')]
            elif quantity == 'ga_expense':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '9.')]
            elif quantity == 'oper_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '8.')]
            elif quantity == 'operprofit_bprovisn':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '10.')]
            elif quantity == 'asset':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.')]
            elif quantity == 'equity':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.II.')]
            elif quantity == 'liability':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.')]
            elif quantity == 'deposit_sbv':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.1.')]
            elif quantity == 'deposit_inst':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.2.')]
            elif quantity == 'deposit_customer':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.3.')]
            elif quantity == 'gov_funding':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.5.')]
            elif quantity == 'paper_issued':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.6.')]
            elif quantity == 'substd':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '4.3.')]
            elif quantity == 'doubtful':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '4.4.')]
            elif quantity == 'baddebt':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '4.5.')]
            elif quantity == 'loans':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '4.')]
            elif quantity == 'provsn_inst':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('bs', 'A.I.3.3.')]
            elif quantity == 'provsn_customer':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('bs', 'A.I.6.2.')]
            elif quantity == 'provsn_charge':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '11.')]
            elif quantity == 'net_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '14.')]
            elif quantity == 'cash':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.1.')]
            elif quantity == 'avaisale_sec':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.8.1.')]
            elif quantity == 'feecomisn_expense':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '2.2')]
            elif quantity == 'other_expense':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '6.2.')]
            elif quantity == 'feecomisn_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '2.1')]
            elif quantity == 'gainlossfxgold':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '3.')]
            elif quantity == 'gainlosstrading':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '4.')]
            elif quantity == 'gainlossinst':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '5.')]
            elif quantity == 'other_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '6.1.')]
            elif quantity == 'dividend_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '7.')]
            elif quantity == 'cur_deposit':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '10.1.')]
            elif quantity == 'saving_deposit':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '10.3.')]
            elif quantity == 'deposit':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('fn', '10.')]
            else:
                pass


# replace 0 values with 1000 VND to avoid 0 denominator
df = df.loc[~(df==0).all(axis=1)] # remove no-data companies first
df = df.replace(to_replace=0, value=1e3)


df['asset_'] = df['asset']
df['equity_'] = df['equity']
df['income'] = df['int_income'] + df['feecomisn_income']\
               + df['gainlossfxgold'] + df['gainlosstrading']\
               + df['gainlossinst'] + df['other_income']\
               + df['dividend_income']
# remove negative revenue companies
df = df.loc[df['income']>0]

df['nim'] = df['net_int_income'] / (df['balnce_sbv']+df['balnce_inst']
                                    + df['trading_sec']+df['loans_customer']
                                    + df['invst_sec'])
df['yoea'] = df['int_income'] / (df['balnce_sbv']+df['balnce_inst']
                                 + df['trading_sec']+df['loans_customer']
                                 + df['invst_sec'])
df['(-)cof'] = -df['int_expense'] / (df['deposit_sbv']+df['deposit_inst']
                                     + df['deposit_customer']+df['gov_funding']
                                     + df['paper_issued'])
df['nonintinc/intinc'] = df['net_feecomisn_income'] / df['net_int_income']
df['(-)cost/income'] = -df['ga_expense'] / df['oper_income']
df['preprovsnroa'] = df['operprofit_bprovisn'] / df['asset']
df['preprovsnroe'] = df['operprofit_bprovisn'] / df['equity']
df['equity/lib'] = df['equity'] / df['liability']
df['equity/loans'] = df['equity'] / df['loans']
df['equity/asset'] = df['equity'] / df['asset']
df['(-)ltd'] = -(df['balnce_inst'] + df['loans_customer']) \
               / (df['deposit_inst'] + df['deposit_customer'])
df['(-)npl'] = -(df['substd'] + df['doubtful'] + df['baddebt']) / df['loans']
df['reserve/npl'] = (df['provsn_customer'] + df['provsn_inst']) \
                    / (df['substd'] + df['doubtful'] + df['baddebt'])
df['reserve/loans'] = (df['provsn_customer'] + df['provsn_inst']) / df['loans']
df['provsn/loans'] = df['provsn_charge'] / df['loans']
df['roe'] = df['net_income'] / df['equity']
df['roa'] = df['net_income'] / df['asset']
df['liquidity'] = (df['cash'] + df['balnce_sbv'] + df['balnce_inst']
                   + df['trading_sec'] + df['avaisale_sec']) / df['liability']
df['equity/asset'] = df['equity'] / df['asset']
df['(-)cir'] = -(df['int_expense'] + df['feecomisn_expense']
                 + df['other_expense'] + df['ga_expense'])\
               / (df['int_income'] + df['feecomisn_income']
                  + df['gainlossfxgold'] + df['gainlosstrading']
                  + df['gainlossinst'] + df['other_income']
                  + df['dividend_income'])
df['(-)lib/equity'] = -df['liability'] / df['equity']
df['loans/asset'] = df['loans'] / df['asset']
df['(-)gaindex'] = -df['ga_expense'] / df['asset']
df['(-)operexp/income'] = -(df['ga_expense'] + df['other_expense']) \
                          / (df['int_income'] + df['feecomisn_income']
                             + df['gainlossfxgold'] + df['gainlosstrading']
                             + df['gainlossinst'] + df['other_income']
                             + df['dividend_income'])
df['lroa'] = (df['cash'] + df['balnce_sbv'] + df['balnce_inst']
              + df['trading_sec'] + df['avaisale_sec']) / df['asset']
df['dgc'] = (df['cash'] + df['balnce_sbv'] + df['balnce_inst']
             + df['trading_sec'] + df['avaisale_sec']) \
            / (df['deposit_sbv'] + df['deposit_inst']
               + df['deposit_customer'])
df['rold'] = df['loans'] / (df['deposit_sbv']
                            + df['deposit_inst']
                            + df['deposit_customer'])
df['efficiency'] = (df['net_int_income'] + df['net_feecomisn_income'])\
                   / (df['feecomisn_expense'] + df['other_expense']
                      + df['ga_expense'])
df['casa'] = (df['cur_deposit'] + df['saving_deposit']) / df['deposit']
df['curdep_ratio'] = df['cur_deposit'] / df['deposit']
df['savingdep_ratio'] = df['saving_deposit'] / df['deposit']
df['net_int/loans'] = df['net_int_income'] / df['loans']
df['int/loans'] = df['int_income'] / df['loans']

df = df.drop(columns=quantities)
df.sort_index(axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

quantities_new = df.columns.to_list()

df.dropna(inplace=True, how='all')

kmeans_index = pd.Index(['bank'])
kmeans = pd.DataFrame(index = kmeans_index, columns = periods)
labels = pd.DataFrame(index = kmeans_index, columns = periods)
centers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_tickers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_coord = pd.DataFrame(index = kmeans_index, columns = periods)

for year, quarter in zip(years, quarters):
    # cross section
    tickers = fa.fin_tickers(True)['bank']
    try:
        df_xs = df.loc[(year, quarter, tickers), :]
    except KeyError:
        continue
    df_xs.dropna(axis=0, how='any', inplace=True)

    for quantity in quantities_new:
        # remove outliers (Interquartile Range Method)
        ## (have to ensure symmetry)
        df_xs_median = df_xs.loc[:, quantity].median()
        df_xs_75q = df_xs.loc[:, quantity].quantile(q=0.75)
        df_xs_25q = df_xs.loc[:, quantity].quantile(q=0.25)
        cut_off = (df_xs_75q - df_xs_25q) * 1.5
        for ticker in df_xs.index.get_level_values(2):
            df_xs.loc[(year,quarter,ticker), quantity] \
                = max(df_xs.loc[(year,quarter,ticker),
                                quantity], df_xs_25q-cut_off)
            df_xs.loc[(year,quarter,ticker), quantity] \
                = min(df_xs.loc[(year,quarter,ticker),
                                quantity], df_xs_75q+cut_off)

        # standardize to mean=0
        df_xs_mean = df_xs.loc[:, quantity].mean()
        for ticker in df_xs.index.get_level_values(2):
            df_xs.loc[(year,quarter,ticker), quantity] \
                = (df_xs.loc[(year,quarter,ticker),
                             quantity]
                   - df_xs_mean)

        # standardize to range (-1,1)
        df_xs_min = df_xs.loc[:, quantity].min()
        df_xs_max = df_xs.loc[:, quantity].max()
        if df_xs_max == df_xs_min:
            df_xs.drop(columns=quantity, inplace=True)
        else:
            for ticker in df_xs.index.get_level_values(2):
                df_xs.loc[(year,quarter,ticker), quantity] \
                    = -1 \
                      + (df_xs.loc[(year,quarter,ticker),
                                   quantity]
                      - df_xs_min) / (df_xs_max-df_xs_min) * 2

    # Kmeans algorithm
    kmeans.loc['bank', f'{year}q{quarter}'] \
        = KMeans(n_clusters=centroids,
                 init='k-means++',
                 n_init=10,
                 max_iter=1000,
                 tol=1e-6,
                 random_state=1)\
        .fit(df_xs.dropna(axis=0, how='any'))

    kmeans_tickers.loc['bank', f'{year}q{quarter}'] \
        = df_xs.index.get_level_values(2).tolist()

    kmeans_coord.loc['bank', f'{year}q{quarter}'] \
        = df_xs.values

    labels.loc['bank', f'{year}q{quarter}'] \
        = kmeans.loc['bank', f'{year}q{quarter}'].labels_.tolist()

    centers.loc['bank', str(year) + 'q' + str(quarter)] \
        = kmeans.loc['bank', f'{year}q{quarter}']\
        .cluster_centers_.tolist()

del df_xs # for memory saving

radius_centers = pd.DataFrame(index=kmeans_index, columns=periods)
for col in range(centers.shape[1]):
    if centers.iloc[0,col] is None:
        radius_centers.iloc[0,col] = None
    else:
        distance = np.zeros(centroids)
        for center in range(centroids):
            # origin at (-1,-1,-1,...) whose dimension varies by PCA
            distance[center] = ((np.array(centers.iloc[0,col][center])
                                 - (-1))**2).sum()**(1/2)
        radius_centers.iloc[0,col] = distance

center_scores = pd.DataFrame(index=kmeans_index, columns=periods)
for col in range(centers.shape[1]):
    if radius_centers.iloc[0,col] is None:
        center_scores.iloc[0,col] = None
    else:
        center_scores.iloc[0,col] \
            = rankdata(radius_centers.iloc[0,col])
        for n in range(1, centroids+1):
            center_scores.iloc[0,col] = \
                np.where(center_scores.iloc[0,col]==n,
                         100/(centroids+1)*n,
                         center_scores.iloc[0,col])

radius_tickers = pd.DataFrame(index=kmeans_index, columns=periods)
for col in range(labels.shape[1]):
    if labels.iloc[0,col] is None:
        radius_tickers.iloc[0,col] = None
    else:
        distance = np.zeros(len(labels.iloc[0,col]))
        for ticker in range(len(labels.iloc[0,col])):
            # origin at (-1,-1,-1,...) whose dimension varies by PCA
            distance[ticker] \
                = (((np.array(kmeans_coord.iloc[0,col][ticker]))
                    - (-1))**2).sum()**(1/2)
        radius_tickers.iloc[0,col] = distance

ticker_raw_scores = pd.DataFrame(index=kmeans_index, columns=periods)#not used
for col in range(labels.shape[1]):
    if labels.iloc[0,col] is None:
        ticker_raw_scores.iloc[0,col] = None
    else:
        raw = np.zeros(len(labels.iloc[0,col]))
        for n in range(len(labels.iloc[0,col])):
            raw[n] = center_scores.iloc[0,col][labels.iloc[0,col][n]]
        ticker_raw_scores.iloc[0,col] = raw

ticker_scores = pd.DataFrame(index=kmeans_index, columns=periods)
for col in range(radius_tickers.shape[1]):
    if radius_tickers.iloc[0,col] is None:
        ticker_scores.iloc[0,col] = None
    else:
        min_ = min(radius_centers.iloc[0, col])
        max_ = max(radius_centers.iloc[0, col])
        range_ = max_ - min_
        f = interp1d(np.sort(np.append(radius_centers.iloc[0,col],
                                       [min_-range_/(centroids-1),
                                        max_+range_/(centroids-1)])),
                     np.sort(np.append(center_scores.iloc[0,col],
                                       [0,100])),
                     kind='linear', bounds_error=False, fill_value=(0,100))
        ticker_scores.iloc[0,col] = f(radius_tickers.iloc[0,col])
        for n in range(len(ticker_scores.iloc[0,col])):
            ticker_scores.iloc[0, col][n] \
                = int(ticker_scores.iloc[0, col][n])


result_table = pd.DataFrame(index=pd.Index(tickers, name='ticker'))
for period in periods:
    try:
        for n in range(len(kmeans_tickers.loc['bank',period])):
            result_table.loc[
                 kmeans_tickers.loc['bank', period][n], period] \
                = ticker_scores.loc['bank', period][n]
    except TypeError:
        continue

#==============================================================================

component_filename = 'component_table_bank'
def export_component_table():
    global destination_dir
    global df
    df.to_csv(join(destination_dir, component_filename+'.csv'))

export_component_table()
df = pd.read_csv(join(destination_dir, component_filename+'.csv'),
                 index_col=['year','quarter','ticker'])


result_filename = 'result_table_bank'
def export_result_table():
    global destination_dir
    global result_table
    result_table.to_csv(join(destination_dir, result_filename+'.csv'))

export_result_table()
result_table = pd.read_csv(join(destination_dir, result_filename+'.csv'),
                           index_col=['ticker'])


def graph_crash(benchmark:float,
                period:str,
                exchange:str='HOSE'):
    crash = ta.crash(benchmark, 'bank', exchange)
    compare_rs(crash[period])


def compare_industry(tickers:list):

    df.dropna(axis=0, how='all', inplace=True)
    median = df.groupby(axis=0, level=[0,1]).median()
    for ticker in tickers:
        # to avoid cases of missing data right at the first period, result in mis-shaped
        quantities = pd.DataFrame(np.zeros_like(median),
                                  index=median.index,
                                  columns=median.columns)
        ref_table = df.xs(ticker, axis=0, level=2)
        quantities = pd.concat([quantities, ref_table], axis=0)
        quantities = quantities.groupby(level=[0,1], axis=0).sum()

        comparison = pd.concat([quantities, median], axis=1, join='outer',
                               keys=[ticker, 'median'])

        periods = [f'{q[0]}q{q[1]}' for q in comparison.index]
        variables \
            = comparison.columns.get_level_values(1).drop_duplicates().to_numpy()

        chartsperrow = 6
        rowsrequired = int(np.ceil(len(variables)/chartsperrow))
        variables = np.append(variables,
                              [None]*(chartsperrow * rowsrequired - len(variables)))
        variables = np.reshape(variables, (rowsrequired, chartsperrow))
        fig, ax = plt.subplots(rowsrequired, chartsperrow,
                               figsize=(chartsperrow*4,rowsrequired*3),
                               tight_layout=True)
        for row in range(rowsrequired):
            for col in range(chartsperrow):
                w = 0.35
                l = np.arange(len(periods))  # the label locations
                if variables[row,col] is None:
                    ax[row, col].axis('off')
                else:
                    ax[row,col].bar(l-w/2,
                                    quantities.iloc[:, row*chartsperrow + col],
                                    width=w, label=ticker,
                                    color='tab:orange', edgecolor='black')
                    ax[row,col].bar(l+w/2, median.iloc[:, row*chartsperrow + col],
                                    width=w, label='Industry\'s Average',
                                    color='tab:blue', edgecolor='black')
                    plt.setp(ax[row,col].xaxis.get_majorticklabels(),
                             rotation=45)
                    ax[row,col].set_xticks(l)
                    ax[row,col].set_xticklabels(periods, fontsize=7)
                    ax[row, col].set_yticks([])
                    ax[row,col].set_autoscaley_on(True)
                    ax[row,col].set_title(variables[row,col], fontsize=9)

        fig.suptitle(f'{ticker}\n'
                     f'Comparison With The Industry\'s Average \n'
                     f'All {len(quantities_new)} Variables In Use \n',
                     fontweight='bold', color='darkslategrey',
                     fontfamily='Times New Roman', fontsize=18)
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left',
                   bbox_to_anchor=(0.01, 0.98), ncol=2, fontsize=10,
                   markerscale=0.7)
        plt.savefig(join(destination_dir, f'{ticker}_compare_industry.png'))


def compare_rs(tickers: list):

    global result_filename
    rs_file = join(dirname(realpath(__file__)), 'research_rating.xlsx')
    rs_rating = pd.read_excel(rs_file, sheet_name='summary',
                              index_col='ticker', engine='openpyxl')

    def scoring(rating: str) -> int:
        mapping = {'AAA': 95, 'AA': 85, 'A': 77.5,
                   'BBB': 72.5, 'BB': 67.5, 'B': 62.5,
                   'CCC': 57.5, 'CC': 52.5, 'C': 47.5,
                   'DDD': 40, 'DD': 30, 'D': 20}
        try:
            return mapping[rating]
        except KeyError:
            return np.nan

    rs_rating = rs_rating.applymap(scoring)
    for i in range(rs_rating.shape[0]):
        for j in range(1, rs_rating.shape[1]-1):
            before = rs_rating.iloc[i, j-1]
            after = np.nan
            k = 0
            if not np.isnan(before) and np.isnan(rs_rating.iloc[i, j]):
                while np.isnan(after) and j+k<rs_rating.shape[1]-1:
                    k += 1
                    after = rs_rating.iloc[i, j+k]
                rs_rating.iloc[i, j] = before + (after-before)/(k+1)

    model_file = join(dirname(realpath(__file__)),
                      'result', result_filename+'.csv')
    model_rating = pd.read_csv(model_file, index_col='ticker')

    for ticker in tickers:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            periods = [q for q in model_rating.columns]
            w = 0.35
            xloc = np.arange(len(periods))  # the label locations
            ax.bar(xloc - w / 2, model_rating.loc[ticker, :],
                   width=w, label='K-Means',
                   color='tab:blue', edgecolor='black')
            ax.bar(xloc + w / 2, rs_rating.loc[ticker, :],
                   width=w, label='Research\'s Rating',
                   color='tab:gray', edgecolor='black')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.set_xticks(xloc)
            ax.set_xticklabels(periods, fontsize=11)

            ax.set_yticks(np.array([0, 25, 50, 75, 100]))
            ax.tick_params(axis='y', labelcolor='black', labelsize=11)

            Acolor = 'green'
            Bcolor = 'olivedrab'
            Ccolor = 'darkorange'
            Dcolor = 'firebrick'

            ax.axhline(100, ls='--', linewidth=0.5, color=Acolor)
            ax.axhline(75, ls='--', linewidth=0.5, color=Bcolor)
            ax.axhline(50, ls='--', linewidth=0.5, color=Ccolor)
            ax.axhline(25, ls='--', linewidth=0.5, color=Dcolor)
            ax.fill_between([-0.4, xloc[-1] + 0.4], 100, 75,
                            color=Acolor, alpha=0.2)
            ax.fill_between([-0.4, xloc[-1] + 0.4], 75, 50,
                            color=Bcolor, alpha=0.25)
            ax.fill_between([-0.4, xloc[-1] + 0.4], 50, 25,
                            color=Ccolor, alpha=0.2)
            ax.fill_between([-0.4, xloc[-1] + 0.4], 25, 0,
                            color=Dcolor, alpha=0.2)

            plt.xlim(-0.6, xloc[-1] + 0.6)

            ax.set_ylim(top=110)
            midpoints = np.array([87.5, 62.5, 37.5, 12.5]) / 110
            labels = ['Group A', 'Group B', 'Group C', 'Group D']
            colors = [Acolor, Bcolor, Ccolor, Dcolor]
            for loc in zip(midpoints, labels, colors):
                ax.annotate(loc[1],
                            xy=(-0.1, loc[0]),
                            xycoords='axes fraction',
                            textcoords="offset points",
                            xytext=(0, -5),
                            ha='center', va='bottom',
                            color=loc[2], fontweight='bold',
                            fontsize='large')
            ax.legend(loc='best', framealpha=5)
            ax.margins(tight=True)
            plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9)
            ax.set_title(ticker + '\n' + "Comparison with Research's Rating",
                         fontsize=15, fontweight='bold', color='darkslategrey',
                         fontfamily='Times New Roman')
            plt.savefig(join(destination_dir, f'{ticker}_compare_rs.png'))
        except KeyError:
            print(f'{ticker} has KeyError')


def mlist_group(year:int, quarter:int) -> dict:

    global result_filename
    file = join(dirname(realpath(__file__)),
                'result', result_filename+'.csv')
    table = pd.read_csv(file, index_col='ticker')

    series = table[str(year) + 'q' + str(quarter)]
    mlist = internal.mlist('all')
    ticker_list = fa.fin_tickers(True)['bank']
    # some tickers in margin list do not have enough data to run K-Means
    model_tickers = table.index.to_list()
    ticker_list = list(set(ticker_list)
                       .intersection(model_tickers)
                       .intersection(mlist))
    series = series.loc[ticker_list]

    def f(score):
        if score <= 25:
            return 'D'
        elif score <= 50:
            return 'C'
        elif score <= 75:
            return 'B'
        elif score <= 100:
            return 'A'
        else:
            return np.nan

    series = series.map(f).dropna()
    groups = series.drop_duplicates().to_numpy()
    d = dict()
    for group in groups:
        tickers = series.loc[series==group].index.to_list()
        d[group] = tickers

    return d


execution_time = time.time() - start_time
print(f"The execution time is: {int(execution_time)}s seconds")