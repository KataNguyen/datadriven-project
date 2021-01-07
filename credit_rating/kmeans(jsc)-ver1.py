import numpy as np
import pandas as pd
import sys
import sklearn
from sklearn import cluster
import time

start_time = time.time()

np.set_printoptions(linewidth=np.inf, precision=4, suppress=True, threshold=sys.maxsize)
pd.set_option("display.max_rows", sys.maxsize,
              "display.max_columns", sys.maxsize,
              'display.expand_frame_repr', True)

quantities = ['revenue', 'cogs', 'gross_profit', 'interest',
              'pbt', 'net_income', 'cur_asset', 'cash', 'ar', 'inv',
            'ppe', 'asset', 'liability', 'cur_liability', 'lt_debt', 'equity']
periods = ['2020:Q3', '2020:Q2', '2020:Q1', '2019:Q4', '2019:Q3', '2019:Q2', '2019:Q1']

col = pd.MultiIndex.from_product([periods, quantities], names=['period', 'quantity'])

df = pd.read_excel(r'C:\Users\Admin\Desktop\PhuHung'
                   r'\CreditRating\FA\Raw_Data_FiinPro(JSC).xlsm',
                   sheet_name='Summary',
                   index_col=[0,1], skiprows=1, header=None)
df.columns = col

for period in periods:
    df.loc[:, (period, 'current_ratio')]\
        = df.loc[:, (period, 'cur_asset')] / df.loc[:, (period, 'cur_liability')]
    df.loc[:, (period, 'quick_ratio')] \
        = (df.loc[:, (period, 'cur_asset')] - df.loc[:, (period, 'inv')]) / \
                                          df.loc[:, (period, 'cur_liability')]
    df.loc[:, (period, 'cash_ratio')] \
        = df.loc[:, (period, 'cash')] / df.loc[:, (period, 'cur_liability')]
    df.loc[:, (period, 'wc_turnover')] \
        = df.loc[:, (period, 'revenue')] / (df.loc[:, (period, 'cur_asset')]
                                            - df.loc[:, (period, 'cur_liability')])
    df.loc[:, (period, 'inv_turnover')] \
        = -df.loc[:, (period, 'cogs')] / df.loc[:, (period, 'inv')]
    df.loc[:, (period, 'ar_turnover')] \
        = df.loc[:, (period, 'revenue')] / df.loc[:, (period, 'ar')]
    df.loc[:, (period, 'ppe_turnover')] \
        = df.loc[:, (period, 'revenue')] / df.loc[:, (period, 'ppe')]
    df.loc[:, (period, 'liability/asset')] \
        = df.loc[:, (period, 'liability')] / df.loc[:, (period, 'asset')]
    df.loc[:, (period, 'lt_debt/equity')] \
        = df.loc[:, (period, 'lt_debt')] / df.loc[:, (period, 'equity')]
    df.loc[:, (period, 'gross_margin')] \
        = df.loc[:, (period, 'gross_profit')] / df.loc[:, (period, 'revenue')]
    df.loc[:, (period, 'net_margin')] \
        = df.loc[:, (period, 'net_income')] / df.loc[:, (period, 'revenue')]
    df.loc[:, (period, 'roe')] \
        = df.loc[:, (period, 'net_income')] / df.loc[:, (period, 'equity')]
    df.loc[:, (period, 'roa')] \
        = df.loc[:, (period, 'net_income')] / df.loc[:, (period, 'asset')]
    df.loc[:, (period, 'ebit/interest')] \
        = (df.loc[:, (period, 'net_income')] - df.loc[:, (period, 'interest')]) / \
          df.loc[:, (period, 'interest')]

df.drop(labels=quantities, axis=1, level=1, inplace=True)
df.sort_index(level=0, axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

quantities_new = [i for i in list(df.columns.levels[1]) if i not in quantities]

for period in periods:
    for ticker in df.index:
        for quantity in quantities_new:
            if df.loc[ticker, (period, quantity)] in [np.nan, np.inf, -np.inf] \
                    or pd.isna(df.loc[ticker, (period, quantity)]):
                df.loc[ticker, period] = np.nan

df.dropna(inplace=True, how='all')

df_textile = df.xs(key='May, sản xuất trang phục và da giày',
                   axis=0, level=1, drop_level=False)
for i in range(len(df_textile.columns)):
    df_textile.iloc[:,i] = (df_textile.iloc[:,i] - df_textile.iloc[:,i].mean()) / \
                           df_textile.iloc[:,i].std()
    for j in range(len(df_textile.index)):
        df_textile.iloc[j,i] = max(df_textile.iloc[j,i], -3)
        df_textile.iloc[j,i] = min(df_textile.iloc[j,i], 3)
df_textile /= 3

df_food = df.xs(key='Chế biến lương thực thực phẩm, đồ uống, thức ăn chăn nuôi',
                axis=0, level=1, drop_level=False)
for i in range(len(df_food.columns)):
    df_food.iloc[:,i] = (df_food.iloc[:,i] - df_food.iloc[:,i].mean()) / \
                        df_food.iloc[:,i].std()
    for j in range(len(df_food.index)):
        df_food.iloc[j,i] = max(df_food.iloc[j,i], -3)
        df_food.iloc[j,i] = min(df_food.iloc[j,i], 3)
df_food /= 3

df_mining = df.xs(key='Khai khoáng', axis=0, level=1, drop_level=False)
for i in range(len(df_mining.columns)):
    df_mining.iloc[:,i] = (df_mining.iloc[:,i] - df_mining.iloc[:,i].mean()) / \
                          df_mining.iloc[:,i].std()
    for j in range(len(df_textile.index)):
        df_mining.iloc[j,i] = max(df_mining.iloc[j,i], -3)
        df_mining.iloc[j,i] = min(df_mining.iloc[j,i], 3)
df_mining /= 3

df_transport = df.xs(key='Kinh doanh vận tải đường bộ, đường sắt, đường thủy, hàng không',
                     axis=0, level=1, drop_level=False)
for i in range(len(df_transport.columns)):
    df_transport.iloc[:,i] = (df_transport.iloc[:,i] - df_transport.iloc[:,i].mean()) / \
                             df_transport.iloc[:,i].std()
    for j in range(len(df_textile.index)):
        df_transport.iloc[j,i] = max(df_transport.iloc[j,i], -3)
        df_transport.iloc[j,i] = min(df_transport.iloc[j,i], 3)
df_transport /= 3

df_chemical = df.xs(key='SX phân bón, hóa chất cơ bản, hạt nhựa cao su tổng hợp',
                    axis=0, level=1, drop_level=False)
for i in range(len(df_chemical.columns)):
    df_chemical.iloc[:,i] = (df_chemical.iloc[:,i] - df_chemical.iloc[:,i].mean()) / \
                            df_chemical.iloc[:,i].std()
    for j in range(len(df_chemical.index)):
        df_chemical.iloc[j,i] = max(df_chemical.iloc[j,i], -3)
        df_chemical.iloc[j,i] = min(df_chemical.iloc[j,i], 3)
df_chemical /= 3

df_consumer_staple = df.xs('Thương mại hàng tiêu dùng', axis=0, level=1, drop_level=False)
for i in range(len(df_consumer_staple.columns)):
    df_consumer_staple.iloc[:,i] = (df_consumer_staple.iloc[:,i]
                                    - df_consumer_staple.iloc[:,i].mean()) /\
                                   df_consumer_staple.iloc[:,i].std()
    for j in range(len(df_consumer_staple.index)):
        df_consumer_staple.iloc[j,i] = max(df_consumer_staple.iloc[j,i], -3)
        df_consumer_staple.iloc[j,i] = min(df_consumer_staple.iloc[j,i], 3)
df_consumer_staple /= 3

df_entertainment = df.xs('Kinh doanh dịch vụ lưu trú, ăn uống, vui chơi giải trí',
                         axis=0, level=1, drop_level=False)
for i in range(len(df_entertainment.columns)):
    df_entertainment.iloc[:,i] = (df_entertainment.iloc[:,i]
                                  - df_entertainment.iloc[:,i].mean()) /\
                                 df_entertainment.iloc[:,i].std()
    for j in range(len(df_entertainment.index)):
        df_entertainment.iloc[j,i] = max(df_entertainment.iloc[j,i], -3)
        df_entertainment.iloc[j,i] = min(df_entertainment.iloc[j,i], 3)
df_entertainment /= 3

df_construction = df.xs(key='Xây dựng (thi công), xây lắp',
                        axis=0, level=1, drop_level=False)
for i in range(len(df_construction.columns)):
    df_construction.iloc[:,i] \
        = (df_construction.iloc[:,i] - df_construction.iloc[:,i].mean()) /\
          df_construction.iloc[:,i].std()
    for j in range(len(df_construction.index)):
        df_construction.iloc[j,i] = max(df_construction.iloc[j,i], -3)
        df_construction.iloc[j,i] = min(df_construction.iloc[j,i], 3)
df_construction /= 3

df_trade_oil = df.xs(key='Thương mại xăng dầu, ga.',
                     axis=0, level=1, drop_level=False)
for i in range(len(df_trade_oil.columns)):
    df_trade_oil.iloc[:,i] = (df_trade_oil.iloc[:,i] - df_trade_oil.iloc[:,i].mean()) / \
                             df_trade_oil.iloc[:,i].std()
    for j in range(len(df_trade_oil.index)):
        df_trade_oil.iloc[j,i] = max(df_trade_oil.iloc[j,i], -3)
        df_trade_oil.iloc[j,i] = min(df_trade_oil.iloc[j,i], 3)
df_trade_oil /= 3

df_material = df.xs(key='SX vật liệu xây dựng (trừ thép)',
                    axis=0, level=1, drop_level=False)
for i in range(len(df_material.columns)):
    df_material.iloc[:,i] = (df_material.iloc[:,i] - df_material.iloc[:,i].mean()) / \
                            df_material.iloc[:,i].std()
    for j in range(len(df_material.index)):
        df_material.iloc[j,i] = max(df_material.iloc[j,i], -3)
        df_material.iloc[j,i] = min(df_material.iloc[j,i], 3)
df_material /= 3

df_energy_tele = df.xs(key='Sản xuất, phân phối điện, năng lượng, dịch vụ viễn thông ',
                       axis=0, level=1, drop_level=False)
for i in range(len(df_energy_tele.columns)):
    df_energy_tele.iloc[:,i] = (df_energy_tele.iloc[:,i] - df_energy_tele.iloc[:,i].mean()) /\
                               df_energy_tele.iloc[:,i].std()
    for j in range(len(df_energy_tele.index)):
        df_energy_tele.iloc[j,i] = max(df_energy_tele.iloc[j,i], -3)
        df_energy_tele.iloc[j,i] = min(df_energy_tele.iloc[j,i], 3)
df_energy_tele /= 3

df_advertising = df.xs(key='Kinh doanh dịch vụ quảng cáo, tư vấn giám sát, in ấn',
                       axis=0, level=1, drop_level=False)
for i in range(len(df_advertising.columns)):
    df_advertising.iloc[:,i] = (df_advertising.iloc[:,i] - df_advertising.iloc[:,i].mean()) / \
                               df_advertising.iloc[:,i].std()
    for j in range(len(df_advertising.index)):
        df_advertising.iloc[j,i] = max(df_advertising.iloc[j,i], -3)
        df_advertising.iloc[j,i] = min(df_advertising.iloc[j,i], 3)
df_advertising /= 3

df_trade_industrial = df.xs(key='Thương mại hàng công nghiệp ',
                            axis=0, level=1, drop_level=False)
for i in range(len(df_trade_industrial.columns)):
    df_trade_industrial.iloc[:,i] \
        = (df_trade_industrial.iloc[:,i]
           - df_trade_industrial.iloc[:,i].mean()) / \
          df_trade_industrial.iloc[:,i].std()
    for j in range(len(df_trade_industrial.index)):
        df_trade_industrial.iloc[j,i] = max(df_trade_industrial.iloc[j,i], -3)
        df_trade_industrial.iloc[j,i] = min(df_trade_industrial.iloc[j,i], 3)
df_trade_industrial /= 3

df_real_estate = df.xs(key='Kinh doanh BDS và cơ sở hạ tầng',
                       axis=0, level=1, drop_level=False)
for i in range(len(df_real_estate.columns)):
    df_real_estate.iloc[:,i] = (df_real_estate.iloc[:,i] - df_real_estate.iloc[:,i].mean()) / \
                               df_real_estate.iloc[:,i].std()
    for j in range(len(df_real_estate.index)):
        df_real_estate.iloc[j,i] = max(df_real_estate.iloc[j,i], -3)
        df_real_estate.iloc[j,i] = min(df_real_estate.iloc[j,i], 3)
df_real_estate /= 3

df_warehouse = df.xs(key='Kinh doanh kho bãi và các dịch vụ hỗ trợ vận tải',
                     axis=0, level=1, drop_level=False)
for i in range(len(df_warehouse.columns)):
    df_warehouse.iloc[:,i] = (df_warehouse.iloc[:,i] - df_warehouse.iloc[:,i].mean()) / \
                             df_warehouse.iloc[:,i].std()
    for j in range(len(df_warehouse.index)):
        df_warehouse.iloc[j,i] = max(df_warehouse.iloc[j,i], -3)
        df_warehouse.iloc[j,i] = min(df_warehouse.iloc[j,i], 3)
df_warehouse /= 3

df_lumber = df.xs(key='Chế biến gỗ, sản xuất sản phẩm từ gỗ và lâm sản khác',
                  axis=0, level=1, drop_level=False)
for i in range(len(df_lumber.columns)):
    df_lumber.iloc[:,i] \
        = (df_lumber.iloc[:,i] - df_lumber.iloc[:,i].mean()) /\
          df_lumber.iloc[:,i].std()
    for j in range(len(df_lumber.index)):
        df_lumber.iloc[j,i] = max(df_lumber.iloc[j,i], -3)
        df_lumber.iloc[j,i] = min(df_lumber.iloc[j,i], 3)
df_lumber /= 3

df_device = df.xs(key='SX điện tử, máy vi tính quang học, thiết bị viễn thông',
                  axis=0, level=1, drop_level=False)
for i in range(len(df_device.columns)):
    df_device.iloc[:,i] = (df_device.iloc[:,i] - df_device.iloc[:,i].mean()) / \
                          df_device.iloc[:,i].std()
    for j in range(len(df_device.index)):
        df_device.iloc[j,i] = max(df_device.iloc[j,i], -3)
        df_device.iloc[j,i] = min(df_device.iloc[j,i], 3)
df_device /= 3

df_steel = df.xs(key='SX thép',
                 axis=0, level=1, drop_level=False)
for i in range(len(df_steel.columns)):
    df_steel.iloc[:,i] = (df_steel.iloc[:,i] - df_steel.iloc[:,i].mean()) / \
                         df_steel.iloc[:,i].std()
    for j in range(len(df_steel.index)):
        df_steel.iloc[j,i] = max(df_steel.iloc[j,i], -3)
        df_steel.iloc[j,i] = min(df_steel.iloc[j,i], 3)
df_steel /= 3

df_trade_material = df.xs(key='Kinh doanh vật liệu xây dựng',
                          axis=0, level=1, drop_level=False)
for i in range(len(df_trade_material.columns)):
    df_trade_material.iloc[:,i] = (df_trade_material.iloc[:,i]
                                   - df_trade_material.iloc[:,i].mean()) / \
                                  df_trade_material.iloc[:,i].std()
    for j in range(len(df_trade_material.index)):
        df_trade_material.iloc[j,i] = max(df_trade_material.iloc[j,i], -3)
        df_trade_material.iloc[j,i] = min(df_trade_material.iloc[j,i], 3)
df_trade_material /= 3

df_seafood = df.xs(key='Chế biến thủy hải sản',
                   axis=0, level=1, drop_level=False)
for i in range(len(df_seafood.columns)):
    df_seafood.iloc[:,i] = (df_seafood.iloc[:,i] - df_seafood.iloc[:,i].mean()) / \
                           df_seafood.iloc[:,i].std()
    for j in range(len(df_seafood.index)):
        df_seafood.iloc[j,i] = max(df_seafood.iloc[j,i], -3)
        df_seafood.iloc[j,i] = min(df_seafood.iloc[j,i], 3)
df_seafood /= 3

df_mechanical = df.xs(key='Cơ khí, chế tạo MMTB , sản xuất kim loại đúc sẵn',
                      axis=0, level=1, drop_level=False)
for i in range(len(df_mechanical.columns)):
    df_mechanical.iloc[:,i] = (df_mechanical.iloc[:,i]
                               - df_mechanical.iloc[:,i].mean()) / \
                              df_mechanical.iloc[:,i].std()
    for j in range(len(df_mechanical.index)):
        df_mechanical.iloc[j,i] = max(df_mechanical.iloc[j,i], -3)
        df_mechanical.iloc[j,i] = min(df_mechanical.iloc[j,i], 3)
df_mechanical /= 3

df_trade_lumber = df.xs(key='Thương mại hàng nông lâm nghiệp',
                        axis=0, level=1, drop_level=False)
for i in range(len(df_trade_lumber.columns)):
    df_trade_lumber.iloc[:,i] = (df_trade_lumber.iloc[:,i]
                                 - df_trade_lumber.iloc[:,i].mean()) / \
                                df_trade_lumber.iloc[:,i].std()
    for j in range(len(df_trade_lumber.index)):
        df_trade_lumber.iloc[j,i] = max(df_trade_lumber.iloc[j,i], -3)
        df_trade_lumber.iloc[j,i] = min(df_trade_lumber.iloc[j,i], 3)
df_trade_lumber /= 3

df_pharma = df.xs(key='SX thuốc, hóa dược, dược liệu',
                  axis=0, level=1, drop_level=False)
for i in range(len(df_pharma.columns)):
    df_pharma.iloc[:,i] = (df_pharma.iloc[:,i] - df_pharma.iloc[:,i].mean()) / \
                          df_pharma.iloc[:,i].std()
    for j in range(len(df_pharma.index)):
        df_pharma.iloc[j,i] = max(df_pharma.iloc[j,i], -3)
        df_pharma.iloc[j,i] = min(df_pharma.iloc[j,i], 3)
df_pharma /= 3

df_equipment = df.xs(key='Sản xuất thiết bị văn phòng, '
                         'đồ gia dụng, thiết bị giáo dục và trang thiết bị y tế',
                     axis=0, level=1, drop_level=False)
for i in range(len(df_equipment.columns)):
    df_equipment.iloc[:,i] = (df_equipment.iloc[:,i] - df_equipment.iloc[:,i].mean()) / \
                             df_equipment.iloc[:,i].std()
    for j in range(len(df_equipment.index)):
        df_equipment.iloc[j,i] = max(df_equipment.iloc[j,i], -3)
        df_equipment.iloc[j,i] = min(df_equipment.iloc[j,i], 3)
df_equipment /= 3

#df_other = df.xs(key='nan', axis=0, level=1)

sector_list = [df_textile, df_food, df_mining,
               df_transport, df_chemical, df_consumer_staple,
               df_entertainment, df_construction, df_trade_oil,
               df_material, df_energy_tele, df_advertising,
               df_trade_industrial, df_real_estate, df_warehouse,
               df_lumber, df_device, df_steel,
               df_trade_material, df_seafood, df_mechanical,
               df_trade_lumber, df_pharma, df_equipment]

#KMeans Classification:
centroids = 4
kmeans = pd.DataFrame(data=np.zeros((len(sector_list), len(periods))))
for i in range(len(periods)):
    for j in range(len(sector_list)):
        kmeans.iloc[j,i] = sklearn.cluster.KMeans(n_clusters=centroids,
                                                  init='k-means++',
                                                  n_init=100,
                                                  max_iter=10000,
                                                  tol=1e-20,
                                                  random_state=1)\
            .fit(sector_list[j].xs(key=periods[i], axis=1, level=0).dropna(how='all'))

labels = dict()
centers= dict()
for i in range(len(periods)):
    for j in range(len(sector_list)):
        labels.update({(periods[i],
                        sector_list[j].index.get_level_values(level=1)
                        .drop_duplicates().values[0]):
                           pd.DataFrame(data=np.array(kmeans.iloc[j,i].labels_),
                                        index=sector_list[j].xs(key=periods[i],
                                                                axis=1, level=0)
                                        .dropna(how='all').index,
                                        columns=[periods[i]])})
        centers.update({(periods[i],
                         sector_list[j].index.get_level_values(level=1)
                         .drop_duplicates().values[0]):
                            pd.DataFrame(data=np.array(kmeans.iloc[j,i].cluster_centers_),
                                         index=['Group 1', 'Group 2', 'Group 3', 'Group 4'],
                                         columns=sector_list[j].xs(key=periods[i],
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

centers_index = pd.MultiIndex.from_product([periods,
                                            [sector.index.get_level_values(level=1)
                                           .drop_duplicates().values[0]
                                             for sector in sector_list],
                                            ['Group 1', 'Group 2', 'Group 3', 'Group 4']],
                                            names=['period', 'sector', 'group'])
result_centers = pd.concat([centers[key] for key in centers.keys()],
                           axis=0, join='outer')\
    .set_index(centers_index)

with pd.ExcelWriter(r'C:\Users\Admin\Desktop\PhuHung\CreditRating'
                    r'\FA\KMeans_Research_Criteria(JSC).xlsx') \
        as writer:
    result_labels.to_excel(writer, sheet_name='Ticker')
    result_centers.to_excel(writer, sheet_name='Center')

print("The execution time is: %s seconds" %(time.time() - start_time))
