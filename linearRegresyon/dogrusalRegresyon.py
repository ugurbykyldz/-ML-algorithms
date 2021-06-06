# y = ax + b 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("satislar.csv")
data.sample(10)
data.shape
data.columns
data.describe()
data.info()
data.duplicated().sum()
data.isnull().sum()


aylar = data["Aylar"].values.reshape(-1,1)
satislar = data["Satislar"].values.reshape(-1,1)

#scale
sc = StandardScaler()
x = sc.fit_transform(aylar) 
y = sc.fit_transform(satislar)


#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2 , random_state=42)


#linear model

lr = LinearRegression()
lr.fit(x_train, y_train)

#tahmin
tahmin = lr.predict(x_test)

#visualize
x_test_data = pd.DataFrame(data= x_test, index= range(x_test.shape[0]), columns = ["aylar"])
y_test_data = pd.DataFrame(data = y_test, index = range(y_test.shape[0]),columns = ["satis"])
tahmin_data = pd.DataFrame(data = tahmin, index = range(tahmin.shape[0]), columns = ["tahmin"])


x_test_data = x_test_data.sort_index()
y_test_data = y_test_data.sort_index()
tahmin_data = tahmin_data.sort_index()

plt.plot(x_test_data["aylar"], y_test_data["satis"],color = "r", label = "Real", marker = 'o')
plt.plot(x_test_data["aylar"],tahmin_data["tahmin"],color = "b",label = "Predict", marker='o')
plt.legend()
plt.show


























