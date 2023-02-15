import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.random.uniform(-3,3,size=100) #生成一百个随机点
x = x.reshape(-1,1)
y = 2*(x**2) + 4*x + 7 + 2*np.random.normal(0,1,size=100).reshape(-1,1)
plt.scatter(x,y)
plt.show()

#%% 线性回归
linear = LinearRegression()
linear.fit(x,y)
y_p1 = linear.predict(x)

#%%多项式回归
ploy = PolynomialFeatures(degree=2)
ploy.fit(x)
x2 = ploy.transform(x)
linear2 = LinearRegression()
linear2.fit(x2,y)
y_p2 = linear2.predict(x2)