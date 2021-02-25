from request_phs import *
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from os.path import dirname, realpath
import itertools
matplotlib.use('Agg')


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

agg_data = fa.all('ins')  #################### expensive

quantities = ['ca', 'cl', 'cash', 'lib', 'asset', 'lt_loans', 'equity',
              'gprofit_ins', 'revenue_ins', 'net_income']

periods = fa.periods
tickers = fa.tickers('ins')

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
            if quantity == 'ca':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.')]
            elif quantity == 'cl':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                    ('bs', 'B.I.1.')]
            elif quantity == 'cash':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.I.1.')]
            elif quantity == 'lib':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                    ('bs', 'B.I.')]
            elif quantity == 'asset':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'A.')]
            elif quantity == 'lt_loans':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs','B.I.3.7.')]
            elif quantity == 'equity':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('bs', 'B.II.')]
            elif quantity == 'gprofit_ins':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '17.')]
            elif quantity == 'revenue_ins':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '6.')]
            elif quantity == 'net_income':
                df.loc[(year, quarter, ticker), quantity] \
                    = agg_data.loc[(year, quarter, ticker),
                                   ('is', '31.')]
            else:
                pass


del agg_data # for memory savings

df = df.loc[~(df==0).all(axis=1)]
df['cur_ratio'] = df['ca'] / df['cl']
df['cash_ratio'] = df['cash'] / df['cl']
df['(-)lib/asset'] = -df['lib'] / df['asset']
df['(-)lt_loans/equity'] = -df['lt_loans'] / df['equity']
df['gross_margin'] = df['gprofit_ins'] / df['revenue_ins']
df['net_margin'] = df['net_income'] / df['revenue_ins']
df['roe'] = df['net_income'] / df['equity']
df['roa'] = df['net_income'] / df['asset']

df = df.drop(columns=quantities)
df.sort_index(axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

quantities_new = df.columns.to_list()

df.dropna(inplace=True, how='all')

kmeans_index = pd.Index(['insurance'])
kmeans = pd.DataFrame(index = kmeans_index, columns = periods)
labels = pd.DataFrame(index = kmeans_index, columns = periods)
centers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_tickers = pd.DataFrame(index = kmeans_index, columns = periods)
kmeans_coord = pd.DataFrame(index = kmeans_index, columns = periods)

for year, quarter in zip(years, quarters):
    # cross section
    tickers = fa.fin_tickers(True)['ins']
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
    kmeans.loc['insurance', str(year) + 'q' + str(quarter)] \
        = KMeans(n_clusters=centroids,
                 init='k-means++',
                 n_init=10,
                 max_iter=1000,
                 tol=1e-6,
                 random_state=1)\
        .fit(df_xs.dropna(axis=0, how='any'))

    kmeans_tickers.loc['insurance', str(year) + 'q' + str(quarter)] \
        = df_xs.index.get_level_values(2).tolist()

    kmeans_coord.loc['insurance', str(year) + 'q' + str(quarter)] \
        = df_xs.values

    labels.loc['insurance', str(year) + 'q' + str(quarter)] \
        = kmeans.loc['insurance', str(year) + 'q' + str(quarter)].labels_.tolist()

    centers.loc['insurance', str(year) + 'q' + str(quarter)] \
        = kmeans.loc['insurance', str(year) + 'q' + str(quarter)]\
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
        for n in range(len(kmeans_tickers.loc['insurance',period])):
            result_table.loc[
                 kmeans_tickers.loc['insurance', period][n], period] \
                = ticker_scores.loc['insurance', period][n]
    except TypeError:
        continue

#==============================================================================

component_filename = 'component_table_ins'
def export_component_table():
    global destination_dir
    global df
    df.to_csv(join(destination_dir, component_filename+'.csv'))

export_component_table()
df = pd.read_csv(join(destination_dir, component_filename+'.csv'),
                 index_col=['year','quarter','ticker'])

result_filename = 'result_table_ins'
def export_result_table():
    global destination_dir
    global result_table
    result_table.to_csv(join(destination_dir, result_filename+'.csv'))

export_result_table()
result_table = pd.read_csv(join(destination_dir, result_filename+'.csv'),
                           index_col=['ticker'])

def graph_ticker(ticker: str):
    table = pd.DataFrame(index=['credit_score'],
                         columns=periods)

    table.loc['credit_score', periods] \
        = result_table.xs(key=ticker, axis=0, level=1).values

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    ax.set_title(ticker,
                 fontsize=15, fontweight='bold', color='darkslategrey',
                 fontfamily='Times New Roman')

    xloc = np.arange(table.shape[1]) # label locations
    rects = ax.bar(xloc, table.iloc[0], width=0.8,
                   color='tab:blue', label='Credit Score', edgecolor='black')
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
                period:str,
                exchange:str='HOSE'):
    crash_list = ta.crash(benchmark, period, 'ins', exchange)
    for ticker in crash_list:
        try:
            graph_ticker(ticker)
            plt.savefig(join(destination_dir,
                             f'crash_{period}_{ticker}_result.png'),
                        bbox_inches='tight')
        except KeyError:
            continue


def graph_all():
    global tickers
    for ticker in tickers:
        try:
            graph_ticker(ticker)
        except KeyError:
            pass


def breakdown(ticker:str):
    table = df.xs(ticker, axis=0, level=2)
    num_quantities = table.shape[1]
    fig = plt.subplots(num_quantities, 1,
                       figsize=(6,14), sharex=True)
    plt.suptitle(f'Raw Component Movement: {ticker}', x=0.52, ha='center',
                 fontweight='bold', color='darkslategrey',
                 fontfamily='Times New Roman', fontsize=17)
    colors = plt.rcParams["axes.prop_cycle"]()
    for i in range(num_quantities):
        yval = table[table.columns[i]].values
        while len(yval) < len(fa.periods):
            yval = np.insert(yval, 0, np.nan)
        fig[1][i].plot(periods, yval,
                       color=next(colors)["color"])
        fig[1][i].grid(True, which='both', axis='x', alpha=0.6)
        fig[1][i].margins(tight=True)
        fig[1][i].set_yticks([])
        fig[1][i].set_ylabel(table.columns[i], labelpad=1,
                             ha='center', fontsize=8.5)

    plt.subplots_adjust(left=0.05,
                        right=0.98,
                        bottom=0.04,
                        top=0.95,
                        hspace=0.1)
    plt.xticks(rotation=45, fontfamily='Times New Roman', fontsize=11)
    plt.savefig(join(destination_dir, f'{ticker}_components'))


def breakdown_all(exchange:str):
    for ticker in fa.tickers('ins', exchange):
        try:
            breakdown(ticker)
        except KeyError:
            pass

def compare_industry(ticker:str):

    df.dropna(axis=0, how='all', inplace=True)
    median = df.groupby(axis=0, level=[0,1]).median()
    # to avoid cases of missing data right at the first period, result in mis-shaped
    quantities = pd.DataFrame(np.zeros_like(median),
                              index=median.index,
                              columns=median.columns)
    ref_table = df.xs(ticker, axis=0, level=2)
    quantities = pd.concat([quantities, ref_table], axis=0)
    quantities = quantities.groupby(level=[0,1], axis=0).sum()

    comparison = pd.concat([quantities, median], axis=1, join='outer',
                           keys=[ticker, 'median'])

    fig, ax = plt.subplots(2, 4, figsize=(18,6),
                           tight_layout=True)

    periods = [str(q[0]) + 'q' + str(q[1]) for q in comparison.index]
    variables \
        = comparison.columns.get_level_values(1).drop_duplicates().to_numpy()
    variables = np.reshape(variables, (2,4))
    for row in range(2):
        for col in range(4):
            w = 0.35
            l = np.arange(len(periods))  # the label locations
            if variables[row,col] is None:
                ax[row, col].axis('off')
            else:
                ax[row,col].bar(l-w/2, quantities.iloc[:, row*4+col],
                                width=w, label=ticker,
                                color='tab:orange', edgecolor='black')
                ax[row,col].bar(l+w/2, median.iloc[:, row*4+col],
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
        for j in range(1, rs_rating.shape[1] - 1):
            before = rs_rating.iloc[i, j - 1]
            after = np.nan
            k = 0
            if not np.isnan(before) and np.isnan(rs_rating.iloc[i, j]):
                while np.isnan(after):
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
    ticker_list = fa.fin_tickers(True)['ins']
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


# Output results
#graph_all('gics', 1)
#graph_crash(-0.5, 'gics', 1, '2020q3', 'gen', 'HOSE')
#breakdown_all('gen')


execution_time = time.time() - start_time
print(f"The execution time is: {int(execution_time)}s seconds")