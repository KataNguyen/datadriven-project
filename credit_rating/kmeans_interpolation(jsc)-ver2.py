from database_.request_data import *
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Parameters:
centroids = 4
min_tickers = 4
std_bound = 2

start_time = time.time()

np.set_printoptions(linewidth=np.inf, precision=4, suppress=True,
                    threshold=sys.maxsize)
pd.set_option("display.max_rows", sys.maxsize,
              "display.max_columns", sys.maxsize,
              'display.expand_frame_repr', True)
pd.options.mode.chained_assignment = None


agg_data = request_fs() ####################

quantities = ['revenue', 'cogs', 'gross_profit', 'interest',
              'pbt', 'net_income', 'cur_asset', 'cash', 'ar', 'inv',
              'ppe', 'asset', 'liability', 'cur_liability', 'lt_debt',
              'equity']

periods = request_period_list()
tickers = request_ticker_list()
standards = request_industry_standard()

years = list()
quarters = list()
for period in periods:
    years.append(period[:4])
    quarters.append(period[-1])

index = pd.MultiIndex.from_arrays([years*len(tickers),
                                   quarters*len(tickers),
                                   tickers*len(periods)],
                                   names=['year', 'quarter', 'ticker'])\
    .sort_values()
col = pd.Index(quantities, name='quantity')

df = pd.DataFrame(np.zeros((len(index), len(col))), columns=col, index=index)

for year in years:
    for quarter in quarters:
        for ticker in tickers:
            for quantity in quantities:
                try:
                    if quantity == 'revenue':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'net_sales']
                    if quantity == 'cogs':
                        df.loc[(year, quarter, ticker), quantity] \
                            = -agg_data.loc[(year, quarter, ticker),
                                            'cost_of_sales']
                    if quantity == 'gross_profit':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'gross_profit']
                    if quantity == 'interest':
                        df.loc[(year, quarter, ticker), quantity] \
                            = -agg_data.loc[(year, quarter, ticker),
                                            'interest_expenses']
                    if quantity == 'pbt':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                'net_accounting_profit/(loss)_before_tax']
                    if quantity == 'net_income':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'net_profit/(loss)_after_tax']
                    if quantity == 'cur_asset':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'current_assets']
                    if quantity == 'cash':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'cash_and_cash_equivalents']
                    if quantity == 'ar':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'accounts_receivable']
                    if quantity == 'inv':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'inventories']
                    if quantity == 'ppe':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'tangible_fixed_assets']
                    if quantity == 'asset':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'total_assets']
                    if quantity == 'liability':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'liabilities']
                    if quantity == 'cur_liability':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'current_liabilities']
                    if quantity == 'lt_debt':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           'long_term_loans']
                    if quantity == 'equity':
                        df.loc[(year, quarter, ticker), quantity] \
                            = agg_data.loc[(year, quarter, ticker),
                                           "owner's_equity"]
                    else:
                        pass
                except KeyError:
                    continue

del agg_data # for memory savings

df['cur_ratio'] = df['cur_asset'] / df['cur_liability']
df['quick_ratio'] = (df['cur_asset'] - df['inv']) / df['cur_liability']
df['cash_ratio'] = df['cash'] / df['cur_liability']
df['wc_turnover'] = df['revenue'] / (df['cur_asset'] - df['cur_liability'])
df['inv_turnover'] = df['cogs'] / df['inv']
df['ar_turnover'] = df['revenue'] / df['ar']
df['ppe_turnover'] = df['revenue'] / df['ppe']
df['lib/asset'] = -df['liability'] / df['asset']
df['lt_debt/equity'] = -df['lt_debt'] / df['equity']
df['gross_margin'] = df['gross_profit'] / df['revenue']
df['net_margin'] = df['net_income'] / df['revenue']
df['roe'] = df['net_income'] / df['equity']
df['roa'] = df['net_income'] / df['asset']
df['ebit/int'] = (df['pbt'] + df['interest']) / df['interest']

df.drop(columns=quantities, inplace=True)
df.sort_index(axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan)

quantities_new = [i for i in list(df.columns) if i not in quantities]

for year, quarter in zip(years, quarters):
    for ticker in tickers:
        for quantity in quantities_new:
            if df.loc[(year, quarter, ticker), quantity] \
                    in [np.nan, np.inf, -np.inf] \
                    or pd.isna(df.loc[(year, quarter, ticker), quantity]):
                df.loc[(year, quarter, ticker), quantity] = np.nan

df.dropna(inplace=True, how='all')

sector_table = dict()
industry_list = dict()
ticker_list = dict()

ind_standards = list()
ind_levels = list()
ind_names = list()
for standard in standards:
    for level in request_industry_level(standard):
        for industry in request_industry_list(standard, int(level[-1])):
            ind_standards.append(standard)
            ind_levels.append(level)
            ind_names.append(industry)
kmeans_index = pd.MultiIndex.from_arrays([ind_standards,
                                          ind_levels,
                                          ind_names],
                                         names=['standard',
                                                'level',
                                                'industry'])
kmeans = pd.DataFrame(index = kmeans_index, columns = periods)
labels = pd.DataFrame(index = kmeans_index, columns = periods)
centers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_tickers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_coord = pd.DataFrame(index = kmeans_index, columns = periods)

# sector_table['standard_name'] -> return: DataFrame (Full table)
# industry_list[('standard_name', level)] -> return: list (all industries)
# ticker_list[('standard_name', level, 'industry')]
# -> return: list (all companies in a particular industry)

for standard in standards:
    sector_table[standard] = request_industry(standard)
    sector_table[standard].columns \
        = pd.RangeIndex(start=1, stop=sector_table[standard].shape[1]+1)

    for level in sector_table[standard].columns.tolist():
        industry_list[(standard, level)] \
            = sector_table[standard][level].drop_duplicates().tolist()

        for industry in industry_list[standard, level]:
            ticker_list[(standard, level, industry)] \
                = sector_table[standard][level]\
                .loc[sector_table[standard][level]==industry]\
                .index.tolist()

            for year, quarter in zip(years, quarters):
                df_xs = df.loc[(year, quarter, ticker_list[(standard,
                                                            level,
                                                            industry)]), :]
                df_xs.dropna(axis=0, how='any', inplace=True)
                if df_xs.shape[0] < min_tickers:
                    kmeans.loc[
                    (standard, standard + '_l' + str(level), industry), :] \
                        = None
                    labels.loc[
                    (standard, standard + '_l' + str(level), industry), :] \
                        = None
                    centers.loc[
                    (standard, standard + '_l' + str(level), industry), :] \
                        = None
                else:
                    for quantity in quantities_new:
                        df_xs_median = df_xs.loc[:,quantity].median()
                        df_xs_std = df_xs.loc[:,quantity].std()
                        if df_xs_std == 0:
                            continue
                        else:
                            df_xs.loc[:,quantity] = (df_xs.loc[:,quantity] \
                                                     - df_xs_median) \
                                                    / df_xs_std

                        for ticker in df_xs.index.get_level_values(2):
                            df_xs.loc[(year,quarter,ticker), quantity] \
                                = max(df_xs.loc[(year,quarter,ticker),
                                                quantity], -std_bound)
                            df_xs.loc[(year,quarter,ticker), quantity] \
                                = min(df_xs.loc[(year,quarter,ticker),
                                                quantity], std_bound)
                        df_xs /= std_bound

                    # Kmeans algorithm
                    kmeans.loc[(standard,
                                standard+'_l'+str(level),
                                industry),
                               year + 'q' + quarter] \
                        = KMeans(n_clusters=centroids,
                                 init='k-means++',
                                 n_init=100,
                                 max_iter=10000,
                                 tol=1e-20,
                                 random_state=1)\
                        .fit(df_xs.dropna(axis=0, how='any'))

                    kmeans_tickers.loc[(standard,
                                        standard + '_l' + str(level),
                                        industry),
                                       year + 'q' + quarter] \
                        = df_xs.index.get_level_values(2).tolist()

                    kmeans_coord.loc[(standard, standard + '_l' + str(level),
                                     industry),
                                     year + 'q' + quarter] \
                        = df_xs.values

                    labels.loc[(standard,
                                standard + '_l' + str(level),
                                industry),
                                year + 'q' + quarter] \
                        = kmeans.loc[(standard,
                                      standard+'_l'+str(level),
                                      industry),
                                     year + 'q' + quarter].labels_.tolist()

                    centers.loc[(standard,
                                 standard + '_l' + str(level),
                                 industry),
                                year + 'q' + quarter] \
                        = kmeans.loc[(standard,
                                      standard+'_l'+str(level),
                                      industry),
                                     year + 'q' + quarter].cluster_centers_\
                        .tolist()

del df, df_xs

origin = np.array([-1 for i in range(len(quantities_new))])

radius_centers = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(centers.shape[0]):
    for col in range(centers.shape[1]):
        if centers.iloc[row,col] is None:
            radius_centers.iloc[row,col] = None
        else:
            distance = np.zeros(centroids)
            for center in range(centroids):
                distance[center] = ((np.array(centers.iloc[row,col][center])
                                     - origin)**2).sum()**(1/2)
            radius_centers.iloc[row,col] = distance

center_scores = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(centers.shape[0]):
    for col in range(centers.shape[1]):
        if radius_centers.iloc[row,col] is None:
            center_scores.iloc[row,col] = None
        else:
            center_scores.iloc[row,col] \
                = rankdata(radius_centers.iloc[row,col])
            for n in range(1, centroids+1):
                center_scores.iloc[row, col] = \
                    np.where(center_scores.iloc[row,col] == n,
                             2*100/(centroids*2)*n-100/(centroids*2),
                             center_scores.iloc[row,col])

radius_tickers = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(labels.shape[0]):
    for col in range(labels.shape[1]):
        if labels.iloc[row,col] is None:
            radius_tickers.iloc[row,col] = None
        else:
            distance = np.zeros(len(labels.iloc[row,col]))
            for ticker in range(len(labels.iloc[row,col])):
                distance[ticker] \
                    = (((np.array(kmeans_coord.iloc[row,col][ticker]))
                        - origin)**2).sum()**(1/2)
            radius_tickers.iloc[row,col] = distance

ticker_raw_scores = pd.DataFrame(index=kmeans_index, columns=periods)#not used
for row in range(labels.shape[0]):
    for col in range(labels.shape[1]):
        if labels.iloc[row,col] is None:
            ticker_raw_scores.iloc[row,col] = None
        else:
            raw = np.zeros(len(labels.iloc[row,col]))
            for n in range(len(labels.iloc[row,col])):
                raw[n] = center_scores.iloc[row,col][labels.iloc[row,col][n]]
            ticker_raw_scores.iloc[row,col] = raw

class radius_tickers_:
    a = list()
    for row in range(radius_tickers.shape[0]):
        for col in range(radius_tickers.shape[1]):
            if radius_tickers.iloc[row,col] is None:
                continue
            else:
                for i in range(len(radius_tickers.iloc[row,col])):
                    a.append(radius_tickers.iloc[row,col][i])
    min = min(a) ; max = max(a)

ticker_scores = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(radius_tickers.shape[0]):
    for col in range(radius_tickers.shape[1]):
        if radius_tickers.iloc[row,col] is None:
            ticker_scores.iloc[row,col] = None
        else:
            f = interp1d(np.sort(np.append(radius_centers.iloc[row,col],
                                           [radius_tickers_.min-1e-12,
                                            radius_tickers_.max+1e-12])),
                         np.sort(np.append(center_scores.iloc[row,col],
                                           [0,100])),
                         kind='linear')
            ticker_scores.iloc[row,col] = f(radius_tickers.iloc[row,col])
            for n in range(len(ticker_scores.iloc[row,col])):
                ticker_scores.iloc[row, col][n] \
                    = int(ticker_scores.iloc[row, col][n])

ind_standards = list()
ind_levels = list()
ind_names = list()
ind_tickers = list()
ind_periods = list()
for standard in standards:
    for level in request_industry_level(standard):
        for industry in request_industry_list(standard, int(level[-1])):
            for period in periods:
                try:
                    if isinstance(
                            kmeans_tickers.loc[
                                (standard,level,industry),period
                            ], str
                    ) is True:
                        ind_standards.append(standard)
                        ind_levels.append(level)
                        ind_names.append(industry)
                        ind_tickers.append(
                            kmeans_tickers.loc[
                                (standard,level,industry),period
                            ]
                        )
                        ind_periods.append(period)
                    else:
                        for ticker in kmeans_tickers.loc[
                            (standard,level,industry),period]:
                            ind_standards.append(standard)
                            ind_levels.append(level)
                            ind_names.append(industry)
                            ind_tickers.append(ticker)
                            ind_periods.append(period)

                except TypeError:
                    continue

result_index = pd.MultiIndex.from_arrays([ind_standards,
                                          ind_levels,
                                          ind_names,
                                          ind_tickers,
                                          ind_periods],
                                         names=['standard',
                                                'level',
                                                'industry',
                                                'ticker',
                                                'period'])

result_table = pd.DataFrame(index=result_index, columns=['credit_score'])
for standard in standards:
    for level in request_industry_level(standard):
        for industry in request_industry_list(standard, int(level[-1])):
            for period in periods:
                try:
                    for n in range(len(kmeans_tickers.loc[
                                           (standard,level,industry),period])):
                        result_table.loc[
                            (standard,
                             level,
                             industry,
                             kmeans_tickers.loc[
                                 (standard, level, industry), period][n],
                             period)] \
                            = ticker_scores.loc[(standard,level,industry),
                                                period][n]
                except TypeError:
                    continue

result_table = result_table.unstack(level=4)
result_table.columns = result_table.columns.droplevel(0)

def result(standard=str, level=int):
    global result_table
    result = result_table.xs(key=standard+'_l'+str(level), axis=0,
                             level=1, drop_level=False)
    return result

returns = request_return() ####################
returns.sort_index(inplace=True) # to improve performance

change_result = pd.DataFrame(index=result_table.index,
                             columns=result_table.columns)
change_result.drop(columns=[result_table.columns[0]], inplace=True)
for col in range(len(result_table.columns)-1):
    change_result.iloc[:,col] \
        = result_table.iloc[:,col+1] - result_table.iloc[:,col]

hot_keys = pd.DataFrame(index=change_result.index,
                        columns=change_result.columns)
for standard, level, industry, ticker in change_result.index:
    for period in change_result.columns:
        if change_result.loc[(standard,level,industry,ticker),
                             period] is None \
            or change_result.loc[(standard,level,industry,ticker),
                                 period] == np.nan \
            or returns.loc[ticker,period] is None \
                or returns.loc[ticker,period] == np.nan:
            hot_keys.loc[(standard,level,industry,ticker),
                         period] = np.nan
        if change_result.loc[(standard,level,industry,ticker),
                             period] \
            * returns.loc[ticker,period] < 0:
            hot_keys.loc[(standard,level,industry,ticker),
                         period] = 0
        if change_result.loc[(standard,level,industry,ticker),
                             period] \
            * returns.loc[ticker,period] >= 0:
            hot_keys.loc[(standard,level,industry,ticker),
                         period] = 1

hot_keys.drop(labels=request_financial_ticker(), axis=0, level=3, inplace=True)
hot_keys.drop(labels='N/A', axis=0, level=2, inplace=True)

def accuracy(standard=str, level=int):
    global hot_keys
    a = hot_keys.xs((standard,standard+'_l'+str(level))).sum().sum() \
        / hot_keys.xs((standard,standard+'_l'+str(level))).count().sum()
    return a

tuples = list()
for standard in standards:
    for level in request_industry_level(standard):
        tuples.append((standard,level))

accuracy_table = pd.DataFrame(index=pd.MultiIndex.from_tuples(tuples),
                              columns=['accuracy_rate'])
for standard, level in accuracy_table.index:
    accuracy_table.loc[(standard,level)] = accuracy(standard,int(level[-1]))

price_table = request_price() ####################



def graph_ticker(standard=str, level=int, ticker=str):
    table = pd.DataFrame(index=['credit_score', 'price'],
                                columns=periods)

    table.loc['credit_score', periods] \
        = result_table.xs(key=(standard, standard + '_l' + str(level)),
                          axis=0, level=[0,1])\
        .xs(key=ticker, axis=0, level=1).values

    table.loc['price', periods] \
        = price_table.loc[ticker,:].values

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    ax1.set_title(ticker + '\n' + standard.upper()
                  + ' Level {} Classification'.format(level),
                  fontsize=15, fontweight='bold', color='darkslategrey',
                  fontfamily='Times New Roman')
    ax2 = ax1.twinx()
    ax1.set_xticklabels(ax1.get_xticks(), rotation=45,
                        fontfamily='Times New Roman', fontsize=13)
    ax1.plot(periods, table.iloc[0], color='tab:red', label='Credit Score')
    ax2.plot(periods, table.iloc[1], color='tab:blue', label='Stock Price')
    ax1.plot(np.nan, color='tab:blue', label='Stock Price')
    ax1.set_ylim(top=np.min([table.iloc[0].max()*1.2, 100]))
    ax1.set_xlabel('periods')
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=11)
    ax1.legend(loc='best', framealpha=5)
    ax1.grid(alpha=0.2)

def graph_classification(standard=str):
    global accuracy_table
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ind = np.arange(
        len(accuracy_table.xs(key=standard, axis=0, level=0).index))
    score = accuracy_table.xs(key=standard, axis=0, level=0).values.T[0]
    ax.bar(ind, score, 0.5, color='tab:blue', edgecolor='black')
    level = accuracy_table.xs(key=standard,
                                      axis=0, level=0).index.tolist()
    for i in range(len(level)):
        level[i] = 'Level ' + level[i][-1]
    plt.xticks(ind, level)
    plt.yticks(np.arange(0.5, 0.61, 0.01))
    ax.set_ylim(bottom=0.5)
    plt.xticks(fontfamily='Times New Roman', fontsize=13)
    plt.yticks(fontfamily='Times New Roman', fontsize=13)

    rects = ax.patches

    for rect, sc in zip(rects, score):
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, height+0.001, np.round(sc,3),
                ha='center', va='bottom',
                fontfamily='Times New Roman', fontsize=13)

    standard = standard.upper()
    plt.title('Accuracy rate of {} by level'.format(standard),
              fontfamily='Times New Roman', fontsize=15, fontweight='bold',
              color='midnightblue')

