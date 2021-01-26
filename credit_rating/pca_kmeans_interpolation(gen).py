from request_phs.request_data import *
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
from os.path import dirname, realpath
import itertools
matplotlib.use('Agg')

destination_dir = join(dirname(realpath(__file__)),
                       'kmeans_interpolation(jsc)-ver2_results')

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

agg_data = request_fs_all('gen')  #################### expensive

quantities = ['revenue', 'cogs', 'gross_profit', 'interest',
              'pbt', 'net_income', 'cur_asset', 'cash', 'ar', 'inv',
              'ppe', 'asset', 'liability', 'cur_liability', 'lt_debt',
              'equity']

periods = request_period()
tickers = request_ticker('gen')
standards = request_industry_standard()

years = list()
quarters = list()
for period in periods:
    years.append(int(period[:4]))
    quarters.append(int(period[-1]))

period_tuple = [(year, quarter) for year, quarter in zip(years,quarters)]
inds = [x for x in itertools.product(period_tuple, tickers)]

for i in range(len(inds)):
    inds[i] = inds[i][0] + tuple([inds[i][1]])

index = pd.MultiIndex.from_tuples(inds, names=['year', 'quarter', 'ticker'])
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
df['lib/asset'] = -df['liability'] / df['asset']
df['lt_debt/equity'] = -df['lt_debt'] / df['equity']
df['gross_margin'] = df['gross_profit'] / df['revenue']
df['net_margin'] = df['net_income'] / df['revenue']
df['roe'] = df['net_income'] / df['equity']
df['roa'] = df['net_income'] / df['asset']
df['ebit/int'] = (df['pbt'] + df['interest']) / df['interest']

df.drop(columns=quantities, inplace=True)
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

    for level in sector_table[standard].columns:
        industry_list[(standard, level)] \
            = sector_table[standard][level].drop_duplicates().tolist()

        for industry in industry_list[standard, level]:
            ticker_list[(standard, level, industry)] \
                = sector_table[standard][level]\
                .loc[sector_table[standard][level]==industry]\
                .index.tolist()

            for year, quarter in zip(years, quarters):
                try:
                    df_xs = df.loc[(year, quarter, ticker_list[(standard,
                                                                level,
                                                                industry)]), :]
                except KeyError:
                    continue
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

                    # Principal Component Analysis
                    X = df_xs.values
                    cov_matrix = np.dot(X.T, X) / X.shape[0]
                    cor_matrix = np.zeros(cov_matrix.shape)
                    for row in range(cov_matrix.shape[0]):
                        for col in range(cov_matrix.shape[1]):
                            cor_matrix[row,col] \
                                = cov_matrix[row,col] \
                                  / (cov_matrix[row,row] ** 0.5
                                     * cov_matrix[col,col] ** 0.5)
                    eig_vals, eig_vects = np.linalg.eig(cor_matrix)

                    eig_dict = dict()
                    for i in range(len(eig_vals)):
                        eig_dict[eig_vals[i]] = eig_vects[:,i]

                    eig_vals = np.array(sorted(eig_vals, reverse=True))
                    eig_matrix = np.zeros(eig_vects.shape)
                    for i in range(len(eig_vals)):
                        eig_matrix[:,i] = eig_dict[eig_vals[i]]

                    pc_matrix = np.dot(X,eig_matrix)
                    explained = np.zeros(len(eig_vals))
                    for i in range(len(eig_vals)):
                        explained[i] = eig_vals[:i].sum() / len(eig_vals)
                    pc_matrix = pc_matrix[:, explained <= 0.9]

                    col = ['pc_'+ str(num) for num
                           in np.arange(start=1, stop=pc_matrix.shape[1]+1)]
                    df_xs = pd.DataFrame(data=pc_matrix, columns=col,
                                         index=df_xs.index)

                    # Kmeans algorithm
                    kmeans.loc[(standard,
                                standard+'_l'+str(level),
                                industry),
                               str(year) + 'q' + str(quarter)] \
                        = KMeans(n_clusters=centroids,
                                 init='k-means++',
                                 n_init=100,
                                 max_iter=10000,
                                 tol=1e-9,
                                 random_state=1)\
                        .fit(df_xs.dropna(axis=0, how='any'))

                    kmeans_tickers.loc[(standard,
                                        standard + '_l' + str(level),
                                        industry),
                                       str(year) + 'q' + str(quarter)] \
                        = df_xs.index.get_level_values(2).tolist()

                    kmeans_coord.loc[(standard, standard + '_l' + str(level),
                                     industry),
                                     str(year) + 'q' + str(quarter)] \
                        = df_xs.values

                    labels.loc[(standard,
                                standard + '_l' + str(level),
                                industry),
                                str(year) + 'q' + str(quarter)] \
                        = kmeans.loc[(standard,
                                      standard+'_l'+str(level),
                                      industry),
                                     str(year) + 'q' + str(quarter)].labels_\
                        .tolist()

                    centers.loc[(standard,
                                 standard + '_l' + str(level),
                                 industry),
                                str(year) + 'q' + str(quarter)] \
                        = kmeans.loc[(standard,
                                      standard+'_l'+str(level),
                                      industry),
                                     str(year) + 'q' + str(quarter)]\
                        .cluster_centers_.tolist()

#del df, df_xs

radius_centers = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(centers.shape[0]):
    for col in range(centers.shape[1]):
        if centers.iloc[row,col] is None:
            radius_centers.iloc[row,col] = None
        else:
            distance = np.zeros(centroids)
            for center in range(centroids):
                # origin at (-1,-1,-1,...) whose dimension varies by PCA
                distance[center] = ((np.array(centers.iloc[row,col][center])
                                     - (-1))**2).sum()**(1/2)
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
                center_scores.iloc[row,col] = \
                    np.where(center_scores.iloc[row,col]==n,
                             100/(centroids+1)*n,
                             center_scores.iloc[row,col])

radius_tickers = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(labels.shape[0]):
    for col in range(labels.shape[1]):
        if labels.iloc[row,col] is None:
            radius_tickers.iloc[row,col] = None
        else:
            distance = np.zeros(len(labels.iloc[row,col]))
            for ticker in range(len(labels.iloc[row,col])):
                # origin at (-1,-1,-1,...) whose dimension varies by PCA
                distance[ticker] \
                    = (((np.array(kmeans_coord.iloc[row,col][ticker]))
                        - (-1))**2).sum()**(1/2)
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

ticker_scores = pd.DataFrame(index=kmeans_index, columns=periods)
for row in range(radius_tickers.shape[0]):
    for col in range(radius_tickers.shape[1]):
        if radius_tickers.iloc[row,col] is None:
            ticker_scores.iloc[row,col] = None
        else:
            min_ = min(radius_centers.iloc[row, col])
            max_ = max(radius_centers.iloc[row, col])
            range_ = max_ - min_
            f = interp1d(np.sort(np.append(radius_centers.iloc[row,col],
                                           [min_-range_/(centroids-1),
                                            max_+range_/(centroids-1)])),
                         np.sort(np.append(center_scores.iloc[row,col],
                                           [0,100])),
                         kind='linear', bounds_error=False, fill_value=(0,100))
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
                    if isinstance(kmeans_tickers.loc[
                                      (standard,level,industry),period],
                                  str) is True:
                        ind_standards.append(standard)
                        ind_levels.append(level)
                        ind_names.append(industry)
                        ind_tickers.append(
                            kmeans_tickers.loc[
                                (standard,level,industry),period])
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


def graph_ticker(standard=str, level=int, ticker=str):
    table = pd.DataFrame(index=['credit_score'],
                         columns=periods)

    table.loc['credit_score', periods] \
        = result_table.xs(key=(standard, standard + '_l' + str(level)),
                          axis=0, level=[0,1])\
        .xs(key=ticker, axis=0, level=1).values

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    ax.set_title(ticker + '\n' + standard.upper()
                 + ' Level {} Classification'.format(level),
                 fontsize=15, fontweight='bold', color='darkslategrey',
                 fontfamily='Times New Roman')
    ax.set_ylim(top=110)

    xloc = np.arange(table.shape[1]) # label locations
    rects = ax.bar(xloc, table.iloc[0], width=0.8,
                   color='tab:blue', label='Credit Score', edgecolor='black')
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x()+rect.get_width()/2, height),
                    xytext=(0,3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_xticks(np.arange(len(xloc)))
    ax.set_xticklabels(table.columns.tolist(), rotation=45, x=xloc,
                       fontfamily='Times New Roman', fontsize=13)

    ax.set_yticks(np.array([0,25,50,75,100]))
    ax.tick_params(axis='y', labelcolor='black', labelsize=11)

    Acolor = 'green'
    Bcolor = 'olivedrab'
    Ccolor = 'darkorange'
    Dcolor = 'firebrick'

    ax.axhline(100, ls='--', linewidth=0.5, color=Acolor)
    ax.axhline(75, ls='--', linewidth=0.5, color=Bcolor)
    ax.axhline(50, ls='--', linewidth=0.5, color=Ccolor)
    ax.axhline(25, ls='--', linewidth=0.5, color=Dcolor)
    ax.fill_between([-0.4,xloc[-1]+0.4], 100, 75,
                    color=Acolor, alpha=0.2)
    ax.fill_between([-0.4,xloc[-1]+0.4], 75, 50,
                    color=Bcolor, alpha=0.25)
    ax.fill_between([-0.4,xloc[-1]+0.4], 50, 25,
                    color=Ccolor, alpha=0.2)
    ax.fill_between([-0.4,xloc[-1]+0.4], 25, 0,
                    color=Dcolor, alpha=0.2)

    plt.xlim(-0.6, xloc[-1] + 0.6)
    ax.legend(loc='best', framealpha=5)
    ax.margins(tight=True)
    plt.savefig(join(destination_dir, f'{ticker}_result.png'),
                bbox_inches='tight')


def graph_crash(benchmark=float, segment=str, period=str):
    crash_list = request_crash(benchmark, segment, period)
    for ticker in crash_list:
        try:
            graph_ticker('bics',3, ticker)
        except KeyError:
            continue
    plt.savefig(join(destination_dir, f'crash_{period}_result.png'),
                bbox_inches='tight')


# Output results
result_table.to_csv(join(destination_dir, f'result_table(3centroids).csv'))
graph_crash(-0.5, 'gen', '2020q3')
for ticker in tickers:
    try:
        graph_ticker('gics', 1, ticker)
    except KeyError:
        pass

execution_time = time.time() - start_time
print(f"The execution time is: {int(execution_time)}s seconds")

