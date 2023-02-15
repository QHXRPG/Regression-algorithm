import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.DataFrame(data=[[22460,7326],[11226,4490],[34547,11546],[4851,2396],[5444,2208],[2662,1608],[4549,2035]],
                    index=['北京','辽宁','上海','江西','河南','贵州','陕西'],
                    columns=['人均GDP/元','人均消费水平/元'])
data.index.name='地区'
#%% 1. 人均GDP/元 和 人均消费水平/元 的散点图
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.scatter(data['人均GDP/元'].values, data['人均消费水平/元'].values)
plt.xlabel("人均GDP/元")
plt.ylabel("人均消费水平/元")
plt.show()

#%% 2.计算线性相关系数
corr = data.corr(method='pearson')
print("相关系数为：",corr.iloc[1,0])

#%% 3.求出估计的回归方程
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit((data['人均GDP/元'].values).reshape(-1,1),(data['人均消费水平/元'].values).reshape(-1,1))
print('系数为：',linear.coef_)
print('回归方程为：',f"y={linear.coef_[0,0]}*x")
y_p = linear.predict((data['人均GDP/元'].values).reshape(-1,1))
plt.scatter(data['人均GDP/元'].values, data['人均消费水平/元'].values)
plt.xlabel("人均GDP/元")
plt.ylabel("人均消费水平/元")
plt.plot((data['人均GDP/元'].values).reshape(-1,1),y_p)
plt.show()

#%% 4.预测人均GDP=5000元的人均消费水平
print('预测其人均消费水平为：',linear.predict([[5000]]))   #2278.10656879元