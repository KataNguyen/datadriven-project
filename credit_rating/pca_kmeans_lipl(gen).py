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

agg_data = fa.all('gen')  #################### expensive

quantities = ['revenue', 'cogs', 'gross_profit', 'interest', 'pbt',
              'net_income', 'cur_asset', 'cash', 'ar', 'lt_receivable', 'inv',
              'ppe', 'asset', 'liability', 'cur_liability', 'lt_debt',
              'equity', 'retn_earnings', 'period_retn_earnings', 'lt_invst',
              'depre_tgbl', 'depre_gw', 's_exp', 'ga_exp', 'oprtng_income',
              'cfo', 'st_debt']

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

index = pd.MultiIndex.from_tuples(inds, names=['year', 'quarter', 'ticker'])
col = pd.Index(quantities, name='quantity')

df = pd.DataFrame(columns=col, index=index)

for year, quarter in period_tuple:
    for ticker in tickers:
        for quantity in quantities:
            if quantity == 'revenue':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '3.')]
            elif quantity == 'cogs':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '4.')]
            elif quantity == 'gross_profit':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '5.')]
            elif quantity == 'interest':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                    ('is', '7.1.')]
            elif quantity == 'pbt':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '16.')]
            elif quantity == 'net_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is','18.')]
            elif quantity == 'cur_asset':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.')]
            elif quantity == 'cash':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.1.')]
            elif quantity == 'ar':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.3.')]
            elif quantity == 'lt_receivable':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.II.1.')]
            elif quantity == 'inv':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.4.')]
            elif quantity == 'ppe':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.II.2.')]
            elif quantity == 'asset':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.')]
            elif quantity == 'liability':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.')]
            elif quantity == 'cur_liability':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.1.')]
            elif quantity == 'lt_debt':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.2.8.')]
            elif quantity == 'equity':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.II.')]
            elif quantity == 'retn_earnings':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.II.1.12.')]
            elif quantity == 'period_retn_earnings':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.II.1.12.2.')]
            elif quantity == 'lt_invst':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.II.5.')]
            elif quantity == 'depre_tgbl':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('cfi', 'I.2.')] # tai sao mot so thang <0?
            elif quantity == 'depre_gw':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('cfi', 'I.3.')]
            elif quantity == 's_exp':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                   ('is', '9.')]
            elif quantity == 'ga_exp':
                df.loc[(year, quarter, ticker), quantity] \
                    = -agg_data.loc[(year, quarter, ticker),
                                   ('is', '10.')]
            elif quantity == 'oprtng_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '11.')]
            elif quantity == 'cfo':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('cfi', 'I.')]
            elif quantity == 'st_debt':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.I.1.10.')]
            else:
                pass


del agg_data # for memory savings

# replace 0 values with 1000 VND to avoid 0 denominator
df = df.loc[~(df==0).all(axis=1)] # remove no-data companies first
df = df.replace(to_replace=0, value=1e3)
# remove negative revenue companies
df = df.loc[df['revenue']>0]


df['cash&ppe'] = df['cash'] + df['ppe']
df['equity_'] = df['equity']
df['revenue_'] = df['revenue']
df['cur_ratio'] = df['cur_asset'] / df['cur_liability']
df['quick_ratio'] = (df['cur_asset'] - df['inv']) / df['cur_liability']
df['cash_ratio'] = df['cash'] / df['cur_liability']
df['(-)inv/cur_asset'] = -df['inv'] / df['cur_asset']
df['wc_turnover'] = df['revenue'] / (df['cur_asset'] - df['cur_liability'])
df['inv_turnover'] = df['cogs'] / df['inv']
df['ar_turnover'] = df['revenue'] / df['ar']
df['ppe_turnover'] = df['revenue'] / df['ppe']
df['(-)lib/asset'] = -df['liability'] / df['asset']
df['(-)lt_debt/equity'] = -df['lt_debt'] / df['equity']
df['gross_margin'] = df['gross_profit'] / df['revenue']
df['net_margin'] = df['net_income'] / df['revenue']
df['roe'] = df['net_income'] / df['equity']
df['roa'] = df['net_income'] / df['asset']
df['ebit/int'] = (df['pbt'] + df['interest']) / df['interest']
df['wc/asset'] = (df['cur_asset'] - df['cur_liability']) / df['asset']
df['retn_earning/asset'] = df['retn_earnings'] / df['asset']
df['ebit/asset'] = (df['pbt'] + df['interest']) / df['asset']
df['rev/asset'] = df['revenue'] / df['asset']
df['dsri'] = -df['ar'] / df['revenue']
df['aqi'] = (df['cur_asset'] + df['ppe'] + df['lt_invst']) / df['asset']
df['depi'] = (df['depre_tgbl'] + df['depre_gw']) \
             / (df['ppe'] + df['depre_tgbl'] + df['depre_gw'])
df['sgai'] = -(df['s_exp'] + df['ga_exp']) / df['revenue']
df['lvgi'] = -(df['cur_liability'] + df['lt_debt'] + df['st_debt']) / df['asset']
df['tata'] = -(df['oprtng_income'] - df['cfo']) / df['asset']
df['cfo/debt'] = df['cfo'] / (df['lt_debt'] + df['st_debt'])
df['ebitda/rev'] = (df['pbt']+df['interest']+df['depre_tgbl']+df['depre_gw'])\
                   / df['revenue']
df['ebit/rev'] = (df['pbt'] + df['interest']) / df['revenue']
df['pbt/rev'] = df['pbt'] / df['revenue']
df['roce'] = (df['pbt']+df['interest']) / (df['cur_asset']-df['cur_liability'])
df['roic'] = df['period_retn_earnings'] \
             / (df['lt_debt'] + df['st_debt'] + df['equity'])
df['tr_turnover'] = df['revenue'] / (df['ar'] + df['lt_receivable'])
df['asset_turnover'] = df['revenue'] / df['asset']
df['equity_turnover'] = df['revenue'] / df['equity']
df['(-)lt_debt/asset'] = -df['lt_debt'] / df['asset']
df['(-)debt/equity'] = -(df['st_debt'] + df['lt_debt'])/ df['equity']
df['(-)debt/asset'] = -(df['st_debt'] + df['lt_debt'])/ df['asset']
df['(-)cur_lib/equity'] = -df['cur_liability'] / df['equity']
df['(-)cur_lib/asset'] = -df['cur_liability'] / df['asset']
df['(-)liability/equity'] = -df['liability'] / df['equity']
df['(-)liability/asset'] = -df['liability'] / df['asset']

df = df.drop(columns=quantities)
df.sort_index(axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

quantities_new = df.columns.to_list()

df.dropna(inplace=True, how='all')

ind_standards = list()
ind_levels = list()
ind_names = list()
for standard in standards:
    for level in fa.levels(standard):
        for industry in fa.industries(standard, int(level[-1])):
            ind_standards.append(standard)
            ind_levels.append(level)
            ind_names.append(industry)
kmeans_index = pd.MultiIndex.from_arrays([ind_standards,
                                          ind_levels,
                                          ind_names],
                                         names=['standard',
                                                'level',
                                                'industry'])

benmrks = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans = pd.DataFrame(index = kmeans_index, columns = periods)
labels = pd.DataFrame(index = kmeans_index, columns = periods)
centers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_tickers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_coord = pd.DataFrame(index = kmeans_index, columns = periods)

sector_table = dict()
industry_list = dict()
ticker_list = dict()

# sector_table['standard_name'] -> return: DataFrame (Full table)
# industry_list[('standard_name', level)] -> return: list (all classification)
# ticker_list[('standard_name', level, 'industry')]
# -> return: list (all companies in a particular industry)

for standard in standards:
    sector_table[standard] = fa.classification(standard)
    sector_table[standard].columns \
        = pd.RangeIndex(start=1, stop=sector_table[standard].shape[1]+1)

    for level in sector_table[standard].columns:
        industry_list[(standard, level)] \
            = sector_table[standard][level].drop_duplicates().to_list()

        for industry in industry_list[standard, level]:
            ticker_list[(standard, level, industry)] \
                = sector_table[standard][level]\
                .loc[sector_table[standard][level]==industry]\
                .index.to_list()

            for year, quarter in zip(years, quarters):
                # cross section
                tickers = ticker_list[(standard, level, industry)]
                try:
                    df_xs = df.loc[(year, quarter, tickers), :]
                except KeyError:
                    continue
                df_xs.dropna(axis=0, how='any', inplace=True)
                if df_xs.shape[0] < min_tickers:
                    kmeans.loc[
                    (standard,f'{standard}_l{level}',industry),:] \
                        = None
                    labels.loc[
                    (standard,f'{standard}_l{level}',industry),:] \
                        = None
                    centers.loc[
                    (standard,f'{standard}_l{level}',industry),:] \
                        = None
                else:
                    for quantity in quantities_new:

                        # remove outliers (Interquartile Range Method)
                        ## (have to ensure symmetry)
                        df_xs_median = df_xs.loc[:, quantity].median()
                        df_xs_75q = df_xs.loc[:, quantity].quantile(q=0.75)
                        df_xs_25q = df_xs.loc[:, quantity].quantile(q=0.25)
                        # deal with rare cases in which most tickers = 0, few != 0
                        if df_xs_75q == df_xs_25q == 0:
                            pass
                        else:
                            cut_off = (df_xs_75q - df_xs_25q) * 1.5
                            for ticker in df_xs.index.get_level_values(2):
                                df_xs.loc[(year,quarter,ticker), quantity] \
                                    = max(df_xs.loc[(year,quarter,ticker),
                                                    quantity],
                                          df_xs_25q-cut_off)
                                df_xs.loc[(year,quarter,ticker), quantity] \
                                    = min(df_xs.loc[(year,quarter,ticker),
                                                    quantity],
                                          df_xs_75q+cut_off)

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

                    # PCA algorithm
                    X = df_xs.values
                    benmrk_vector \
                        = np.array([-1 for n in range(df_xs.shape[1])])
                    benmrk_vector \
                        = benmrk_vector.reshape((1, benmrk_vector.shape[0]))
                    X = np.append(benmrk_vector, X, axis=0)
                    PCs = PCA(n_components=0.9).fit_transform(X)

                    idx = df_xs.index.to_list()
                    idx.insert(0, (year, quarter, 'BM_'))
                    idx = pd.MultiIndex.from_tuples(idx)
                    cols = ['PC_'+str(i+1) for i in range(PCs.shape[1])]
                    df_xs = pd.DataFrame(PCs, index=idx, columns=cols)

                    benmrks.loc[(standard, f'{standard}_l{level}', industry),
                                f'{year}q{quarter}'] = PCs[0]

                    # Kmeans algorithm
                    kmeans.loc[(standard, f'{standard}_l{level}',industry),
                               f'{year}q{quarter}'] \
                        = KMeans(n_clusters=centroids,
                                 init='k-means++',
                                 n_init=10,
                                 max_iter=1000,
                                 tol=1e-6,
                                 random_state=1)\
                        .fit(df_xs.dropna(axis=0, how='any'))

                    kmeans_tickers.loc[(standard, f'{standard}_l{level}',
                                        industry), f'{year}q{quarter}'] \
                        = df_xs.index.get_level_values(2).to_list()

                    kmeans_coord.loc[(standard, f'{standard}_l{level}',
                                     industry), f'{year}q{quarter}'] \
                        = df_xs.values

                    labels.loc[(standard, f'{standard}_l{level}', industry),
                               f'{year}q{quarter}'] \
                        = kmeans.loc[(standard, f'{standard}_l{level}',
                                      industry),
                                     f'{year}q{quarter}'].labels_\
                        .tolist()

                    centers.loc[(standard, f'{standard}_l{level}', industry),
                                f'{year}q{quarter}'] \
                        = kmeans.loc[(standard, f'{standard}_l{level}',
                                      industry), f'{year}q{quarter}']\
                        .cluster_centers_.tolist()
                    print(f'Passed:: Standard: {standard.upper()} '
                          f'- Level: {level} -- {industry} '
                          f'-- Year: {year}, Quarter: {quarter}')

del df_xs # for memory saving

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
                                     - np.array(benmrks.iloc[row,col]))
                                    **2).sum()**(1/2)
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
                distance[ticker] \
                    = (((np.array(kmeans_coord.iloc[row,col][ticker]))
                        - np.array(benmrks.iloc[row,col]))**2).sum()**(1/2)
            radius_tickers.iloc[row,col] = distance

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
    for level in fa.levels(standard):
        for industry in fa.industries(standard, int(level[-1])):
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
    for level in fa.levels(standard):
        for industry in fa.industries(standard, int(level[-1])):
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


component_filename = 'component_talbe_gen'
def export_component_table():
    global destination_dir
    global df
    df.to_csv(join(destination_dir, component_filename+'.csv'))

export_component_table()
df = pd.read_csv(join(destination_dir, component_filename+'.csv'),
                 index_col=['year','quarter','ticker'])


result_filename = 'result_table_gen'
def export_result_table():
    global destination_dir
    global result_table
    result_table.to_csv(join(destination_dir, result_filename+'.csv'))

export_result_table()
result_table = pd.read_csv(join(destination_dir, result_filename+'.csv'),
                           index_col=['standard','level','industry','ticker'])


def graph_tickers(tickers: list, standard: str, level: int):
    table = pd.DataFrame(index=['credit_score'],
                         columns=periods)
    for ticker in tickers:
        for period in periods:
            table.loc['credit_score', period] \
                = result_table.loc[
                pd.IndexSlice[standard,f'{standard}_l{level}',:,ticker],
                period].iloc[0]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax.set_title(ticker + '\n' + standard.upper()
                     + ' Level {} Classification'.format(level),
                     fontsize=15, fontweight='bold', color='darkslategrey',
                     fontfamily='Times New Roman')

        xloc = np.arange(table.shape[1]) # label locations
        rects = ax.bar(xloc, table.iloc[0], width=0.8,
                       color='tab:blue',
                       label='Credit Score',
                       edgecolor='black')
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.0f}'.format(height),
                        xy=(rect.get_x()+rect.get_width()/2, height),
                        xytext=(0,3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

        ax.set_xticks(np.arange(len(xloc)))
        ax.set_xticklabels(table.columns.tolist(), rotation=45, x=xloc,
                           fontfamily='Times New Roman', fontsize=11)

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

        ax.set_ylim(top=110)
        midpoints = np.array([87.5, 62.5, 37.5, 12.5])/110
        labels = ['Group A', 'Group B', 'Group C', 'Group D']
        colors = [Acolor, Bcolor, Ccolor, Dcolor]
        for loc in zip(midpoints, labels, colors):
            ax.annotate(loc[1],
                        xy=(-0.1, loc[0]),
                        xycoords='axes fraction',
                        textcoords="offset points",
                        xytext=(0,-5),
                        ha='center', va='bottom',
                        color=loc[2], fontweight='bold',
                        fontsize='large')

        ax.legend(loc='best', framealpha=5)
        ax.margins(tight=True)
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9)
        plt.savefig(join(destination_dir, f'{ticker}_result.png'))


def graph_crash(benchmark:float,
                standard,
                level,
                period:str,
                exchange:str='all'):
    crash = ta.crash(benchmark, 'gen', exchange)
    compare_rs(crash[period], standard, level)


def compare_industry(tickers:list, standard:str, level:int, nperiods:int=12):

    full_list = fa.classification(standard).iloc[:,level-1]
    for ticker in tickers:
        # to avoid cases of missing data right at the first period, result in mis-shaped
        industry = full_list.loc[ticker]
        peers = full_list.loc[full_list == industry].index.tolist()

        table = df.loc[df.index.get_level_values(2).isin(peers)]
        table.dropna(axis=0, how='all', inplace=True)

        median = table.groupby(axis=0, level=[0, 1]).median()
        quantities = pd.DataFrame(np.zeros_like(median),
                                  index=median.index,
                                  columns=median.columns)

        ref_table = table.xs(ticker, axis=0, level=2)
        quantities = pd.concat([quantities, ref_table], axis=0)
        quantities = quantities.groupby(level=[0, 1], axis=0).sum()

        comparison = pd.concat([quantities, median], axis=1, join='outer',
                               keys=[ticker, 'median'])

        periods = [f'{q[0]}q{q[1]}' for q in comparison.index]
        variables \
            = comparison.columns.get_level_values(1)\
            .drop_duplicates().to_numpy()

        chartsperrow = 6
        rowsrequired = int(np.ceil(len(variables) / chartsperrow))
        variables = np.append(variables,
                              [None] * (chartsperrow*rowsrequired
                                        - len(variables)))
        variables = np.reshape(variables, (rowsrequired, chartsperrow))
        fig, ax = plt.subplots(rowsrequired, chartsperrow,
                               figsize=(chartsperrow * 4, rowsrequired * 3),
                               tight_layout=True)
        for row in range(rowsrequired):
            for col in range(chartsperrow):
                w = 0.35
                l = np.arange(len(periods))  # the label locations
                if variables[row, col] is None:
                    ax[row,col].axis('off')
                else:

                    y_quantities = pd.Series([np.nan]*len(periods),
                                             index=periods)
                    y_quantities.iloc[-nperiods:] \
                        = quantities.iloc[-nperiods:, row * chartsperrow + col]
                    ax[row,col].bar(l - w/2, y_quantities,
                                     width=w, label=ticker,
                                     color='tab:orange', edgecolor='black')

                    y_median = pd.Series([np.nan]*len(periods),
                                         index=periods)
                    y_median.iloc[-nperiods:] \
                        = median.iloc[-nperiods:, row * chartsperrow + col]
                    ax[row,col].bar(l + w/2, y_median,
                                     width=w, label='Industry\'s Average',
                                     color='tab:blue', edgecolor='black')

                    ax[row,col].axhline(y=0, color='black', alpha=0.5, lw=1)

                    plt.setp(ax[row, col].xaxis.get_majorticklabels(),
                             rotation=45)
                    ax[row,col].set_xticks(l)
                    ax[row,col].set_xticklabels(periods, fontsize=7)
                    ax[row,col].set_yticks([])
                    ax[row,col].set_autoscaley_on(True)
                    ax[row,col].set_title(variables[row, col], fontsize=9)

        fig.suptitle(f'{ticker}\n'
                     f'Comparison With The Industry\'s Average \n'
                     f'All {len(quantities_new)} Variables In Use \n',
                     fontweight='bold', color='darkslategrey',
                     fontfamily='Times New Roman', fontsize=18)
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left',
                   bbox_to_anchor=(0.01, 0.98), ncol=2, fontsize=10,
                   markerscale=0.7)
        plt.savefig(join(destination_dir, f'{ticker}_compare_industry.png'))


def compare_rs(tickers: list, standard: str, level: int):

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
        for j in range(1, rs_rating.shape[1] - 1):
            before = rs_rating.iloc[i, j - 1]
            after = np.nan
            k = 0
            if not np.isnan(before) and np.isnan(rs_rating.iloc[i, j]):
                while np.isnan(after) and j+k<rs_rating.shape[1]-1:
                    k += 1
                    after = rs_rating.iloc[i, j + k]
                rs_rating.iloc[i, j] = before + (after - before) / (k + 1)

    model_file = join(dirname(realpath(__file__)),
                      'result', result_filename+'.csv')
    model_rating = pd.read_csv(model_file, index_col='ticker')
    model_rating \
        = model_rating.loc[model_rating['standard'] == standard]
    model_rating \
        = model_rating.loc[
        model_rating['level'] == standard + '_l' + str(level)]
    model_rating.drop(columns=['standard', 'level', 'industry'], inplace=True)

    for ticker in tickers:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            periods = [q for q in model_rating.columns]
            w = 0.35
            xloc = np.arange(len(periods))  # the label locations
            ax.bar(xloc - w/2, model_rating.loc[ticker, :],
                   width=w, label='K-Means',
                   color='tab:blue', edgecolor='black')
            ax.bar(xloc + w/2, rs_rating.loc[ticker, :],
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
            pass


def mlist_group(standard:str, level:int, year:int, quarter:int) -> dict:

    global result_filename
    file = join(dirname(realpath(__file__)),
                'result', result_filename+'.csv')
    table = pd.read_csv(file, index_col='ticker')

    table = table.loc[table['standard'] == standard]
    table = table.loc[table['level'] == standard + '_l' + str(level)]
    table.drop(columns=['standard', 'level', 'industry'], inplace=True)

    series = table[str(year) + 'q' + str(quarter)]
    mlist = internal.mlist('all')
    fin_tickers = fa.fin_tickers(False)
    ticker_list = [ticker for ticker in mlist if ticker not in fin_tickers]
    # some tickers in margin list do not have enough data to run K-Means
    model_tickers = table.index.to_list()
    ticker_list = list(set(ticker_list).intersection(model_tickers))
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