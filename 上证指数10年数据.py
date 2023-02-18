import pandas as pd
import numpy as np
data = pd.read_csv("/Users/qiuhaoxuan/Downloads/202302179IzPwYUM/上证指数历史数据2000.csv")
a = pd.read_csv("/Users/qiuhaoxuan/Downloads/book-master/test/week11/上证指数2021-日期-时间-开-高-低-收-成交量-成交额.csv")
date = a.iloc[0].values[0].split()
c=pd.read_csv("上证指数2021.csv")
c=c.drop(labels='Unnamed: 0',axis=1)
time=c['时间'].values

ground_begin = c.groupby(by='日期').apply(lambda x:list(x.开))
for i in range(len(ground_begin)):
    ground_begin.iloc[i] = ground_begin.iloc[i][0]

ground_end = c.groupby(by='日期').apply(lambda x:list(x.收))
for i in range(len(ground_end)):
    ground_end.iloc[i] = ground_end.iloc[i][-1]

ground_max = c.groupby(by='日期').apply(lambda x:list(x.高))
for i in range(len(ground_max)):
    ground_max.iloc[i] = np.max(ground_max.iloc[i])

ground_min = c.groupby(by='日期').apply(lambda x:list(x.低))
for i in range(len(ground_min)):
    ground_min.iloc[i] = np.min(ground_min.iloc[i])

ground_sum = c.groupby(by='日期').apply(lambda x:list(x.成交量))
for i in range(len(ground_sum)):
    ground_sum.iloc[i] = np.sum(ground_sum.iloc[i])

date = data['日期'].values
for i in range(len(date)):
    date[i] = date[i].replace('年','-')
    date[i] = date[i].replace('月', '-')
    date[i] = date[i].replace('日', '')
data.index = date
data = data.drop(labels='日期',axis=1)
data = data.drop(labels='涨跌幅',axis=1)

money_sum = data['交易量'].values
for i in range(len(money_sum)):
    if 'B' in money_sum[i]:
        money_sum[i] = float(money_sum[i].replace('B',''))
    elif 'M' in money_sum[i]:
        money_sum[i] = float(money_sum[i].replace('M',''))/1000
data = data.drop(labels='交易量',axis=1)
data['交易量'] = money_sum

B=10000000
data_2021 = pd.DataFrame()
data_2021['收盘'] =ground_end
data_2021['开盘'] =ground_begin
data_2021['高'] =ground_max
data_2021['低'] =ground_min
data_2021['交易量'] =ground_sum
for i in range(len(data_2021)):
    data_2021['交易量'][i] = float(data_2021['交易量'][i])/B
    data_2021['收盘'][i] = float(data_2021['收盘'][i])
    data_2021['开盘'][i] = float(data_2021['开盘'][i])
    data_2021['高'][i] = float(data_2021['高'][i])
    data_2021['低'][i] = float(data_2021['低'][i])
for i in range(len(data)):
    data['收盘'][i] = float(data['收盘'][i].replace(',',''))
    data['开盘'][i] = float(data['开盘'][i].replace(',',''))
    data['高'][i] = float(data['高'][i].replace(',',''))
    data['低'][i] = float(data['低'][i].replace(',',''))
data = data.iloc[::-1]
data_2000_2021 = pd.concat([data,data_2021.iloc[74:]],axis=0)
data_2000_2021.to_csv('上证指数21年数据.csv')

#%%
data_2000_2021 = pd.read_csv('/Users/qiuhaoxuan/PycharmProjects/数据分析/几种常见回归算法/上证指数21年数据.csv')
data_2000_2021.index=data_2000_2021['Unnamed: 0']
data_2000_2021=data_2000_2021.drop(labels='Unnamed: 0',axis=1)
data2022 = pd.read_csv("/Users/qiuhaoxuan/Desktop/数据集/上证指数2022.csv")
for i in range(len(data2022)):
    data2022['日期'][i] = data2022['日期'][i].replace('/','-')
    data2022['交易量'][i] = data2022['交易量'][i]/10000000
data2022.index=data2022['日期']
data2022=data2022.drop(labels='日期',axis=1)
data_2000_2022 = pd.concat([data_2000_2021,data2022],axis=0)
data_2000_2022.to_csv('上证指数22年数据.csv')

#%%
data_2000_2023 = pd.read_csv('/Users/qiuhaoxuan/Desktop/数据集/上证指数2000_2023.csv')