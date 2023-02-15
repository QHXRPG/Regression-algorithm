import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch.utils.data as Data
data = load_iris()
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.6)
standard = StandardScaler()
x_train,x_test = standard.fit_transform(x_train), standard.transform(x_test)

#%% 方法一
logistic1 = LogisticRegression()
logistic1.fit(x_train,y_train)
y_p = logistic1.predict(x_test)


#%% 方法二
x_train,x_test,y_train,y_test = torch.tensor(x_train),torch.tensor(x_test),torch.tensor(y_train),torch.tensor(y_test)
x_train,x_test = x_train.to(torch.float32),x_test.to(torch.float32)
train_data = Data.TensorDataset(x_train,y_train)
data_loader = Data.DataLoader(dataset=train_data, batch_size=90)
class classifier_linear(nn.Module):
    def __init__(self):
        super(classifier_linear, self).__init__()
        self.model = nn.Sequential(nn.Linear(4,10),
                                   nn.Sigmoid(),
                                   nn.Linear(10,4))
    def forward(self,x):
        return self.model(x)
logistic2 = classifier_linear()
opt = torch.optim.Adam(logistic2.parameters(), lr=0.05)
loss = nn.CrossEntropyLoss()
for i in range(200):
    for j,(x,y) in enumerate(data_loader):
        y_p = logistic2(x)
        l = loss(y_p,y)
        opt.zero_grad()
        l.backward()
        opt.step()


#%%
y_p2 = logistic2(x_test)
y_p2 = torch.argmax(y_p2,axis=1)
accuracy_rate = torch.sum(y_p2==y_test)/len(y_test)
print(f"方法一:{logistic1.score(x_test,y_test)}")
print(f"方法二:{accuracy_rate}")