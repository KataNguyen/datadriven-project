import numpy as np
import pandas as pd
import sys
import sklearn
from sklearn import cluster
import time

start_time = time.time()

np.set_printoptions(linewidth=np.inf,
                    precision=4,
                    suppress=True,
                    threshold=sys.maxsize)
pd.set_option("display.max_rows", sys.maxsize,
              "display.max_columns", sys.maxsize,
              'display.expand_frame_repr', True)

quantities = ['cash', 'asset', 'liability',
              'equity', 'cash_sbv', 'cash_ins',
              'trading_sc',	'loan', 'provision',
              'avai_sc', 'mat_sc', 'deposit',
              'sub_loan', 'doubt_loan', 'bad_loan',
              'pbt', 'net_income', 'int_income',
              'int_exp', 'ser_income', 'ser_exp',
              'fxgold_gainloss', 'trading_gainloss', 'invst_gainloss',
              'other_income', 'other_exp', 'div_income',
              'ga_exp', 'car_']

periods = ['2020:Q3', '2020:Q2', '2020:Q1',
           '2019:Q4', '2019:Q3', '2019:Q2', '2019:Q1']

col = pd.MultiIndex.from_product([periods, quantities],
                                 names=['period', 'quantity'])

df = pd.read_excel(r'C:\Users\Admin\Desktop\PhuHung\CreditRating'
                   r'\FA\Raw_Data_FiinPro(Bank).xlsm', sheet_name='Summary',
                   index_col=[0,1], skiprows=1, header=None)
df.columns = col

for period in periods:
    df.loc[:, (period, 'nim')] \
        = df.loc[:, (period, 'int_income')]/ \
          (df.loc[:, (period, 'cash_ins')]
           + df.loc[:, (period, 'trading_sc')]
           + df.loc[:, (period, 'loan')]
           + df.loc[:, (period, 'avai_sc')]
           + df.loc[:, (period, 'mat_sc')])
    df.loc[:, (period, 'roa')] \
        = df.loc[:, (period, 'net_income')]/ \
          df.loc[:, (period, 'asset')]
    df.loc[:, (period, 'roe')] \
        = df.loc[:, (period, 'net_income')]/\
          df.loc[:, (period, 'equity')]
    df.loc[:, (period, 'npl')] \
        = (df.loc[:, (period, 'sub_loan')]
           + df.loc[:, (period, 'doubt_loan')]
           + df.loc[:, (period, 'bad_loan')])/ \
          df.loc[:, (period, 'loan')]
    df.loc[:, (period, 'bd_provision')] \
        = - df.loc[:, (period, 'provision')]/ \
          df.loc[:, (period, 'loan')]
    df.loc[:, (period, 'liquidity_ratio')] \
        = (df.loc[:, (period, 'cash')]
           + df.loc[:, (period, 'cash_sbv')]
           + df.loc[:, (period, 'cash_ins')]
           + df.loc[:, (period, 'trading_sc')]
           + df.loc[:, (period, 'avai_sc')])/ \
          df.loc[:, (period, 'loan')]
    df.loc[:, (period, 'ld_ratio')] \
        = df.loc[:, (period, 'loan')] / \
          df.loc[:, (period, 'deposit')]
    df.loc[:, (period, 'equity_multi')] \
        = df.loc[:, (period, 'equity')] / \
          df.loc[:, (period, 'asset')]
    df.loc[:, (period, 'car')] = df.loc[:, (period, 'car_')]
    df.loc[:, (period, 'cir')] = - (df.loc[:, (period, 'int_exp')]
                                  - df.loc[:, (period, 'ser_exp')]
                                  - df.loc[:, (period, 'other_exp')]
                                  - df.loc[:, (period, 'ga_exp')])\
                                 / (df.loc[:, (period, 'int_income')]
                                  + df.loc[:, (period, 'ser_income')]
                                  + df.loc[:, (period, 'fxgold_gainloss')]
                                  + df.loc[:, (period, 'trading_gainloss')]
                                  + df.loc[:, (period, 'invst_gainloss')]
                                  + df.loc[:, (period, 'other_income')]
                                  + df.loc[:, (period, 'div_income')])

df.drop(labels=quantities, axis=1, level=1, inplace=True)
df.sort_index(level=0, axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

quantities_new = [i for i in list(df.columns.levels[1]) if i not in quantities]

for period in periods:
    for ticker in df.index:
        for quantity in quantities_new:
            if df.loc[ticker, (period, quantity)] in [np.nan, np.inf, -np.inf]\
                    or pd.isna(df.loc[ticker, (period, quantity)]):
                df.loc[ticker, period] = np.nan

df.dropna(inplace=True, how='all')

for i in range(len(df.columns)):
    df.iloc[:,i] = (df.iloc[:,i] - df.iloc[:,i].mean()) / \
                             df.iloc[:,i].std()
    for j in range(len(df.index)):
        df.iloc[j,i] = max(df.iloc[j,i], -3)
        df.iloc[j,i] = min(df.iloc[j,i], 3)
df /= 3

#KMeans Classification:
centroids = 4
kmeans = pd.DataFrame(data=np.zeros(len(periods)))
for i in range(len(periods)):
    kmeans.iloc[i] = sklearn.cluster.KMeans(n_clusters=centroids,
                                            init='k-means++',
                                            n_init=100,
                                            max_iter=10000,
                                            tol=1e-20,
                                            random_state=1)\
        .fit(df.xs(key=periods[i], axis=1, level=0).dropna(how='all'))

labels = dict()
centers= dict()
for i in range(len(periods)):
    labels.update({(periods[i],
                    df.index.get_level_values(level=1)
                    .drop_duplicates().values[0]):
                       pd.DataFrame(data=np.array(kmeans.iloc[i,0].labels_),
                                    index=df.xs(key=periods[i],
                                                axis=1, level=0)
                                    .dropna(how='all').index,
                                    columns=[periods[i]])})
    centers.update({(periods[i],
                     df.index.get_level_values(level=1)
                     .drop_duplicates().values[0]):
                        pd.DataFrame(data=np.array(kmeans.iloc[i,0]
                                                   .cluster_centers_),
                                     index=['Group 1', 'Group 2',
                                            'Group 3', 'Group 4'],
                                     columns=df.xs(key=periods[i],
                                                   axis=1, level=0)
                                     .dropna(how='all').columns)})

result_labels \
    = pd.concat([labels[key] for key in labels.keys()],
                axis=0, join='outer')\
    .groupby(level=[0,1] ,axis=0)\
    .sum(min_count=1)\
    .sort_index(axis=1)\
    .sort_index(axis=0, level=1)
result_labels.index.set_names(names=['ticker', 'sector'],
                              level=[0,1], inplace=True)

centers_index = pd.MultiIndex\
    .from_product([periods, ['Group 1', 'Group 2', 'Group 3', 'Group 4']],
                  names=['period', 'group'])

result_centers = pd.concat([centers[key] for key in centers.keys()],
                           axis=0, join='outer').set_index(centers_index)

with pd.ExcelWriter(r'C:\Users\Admin\Desktop\PhuHung'
                    r'\CreditRating\FA\KMeans_Research_Criteria(Bank)'
                    r'.version2.xlsx') \
        as writer:
    result_labels.to_excel(writer, sheet_name='Ticker')
    result_centers.to_excel(writer, sheet_name='Center')

print("The execution time is: %s seconds" %(time.time() - start_time))
