# max'um marjini çizer
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


#veri yükleme
veriler =  pd.read_csv("maaslar.csv")

#veri ön işleme
#rastgele 10 ornek
veriler.sample(10)
#veriler hakkında  bilgi ogrenme
veriler.info()
#sutunlara ulaşma
veriler.columns
#eksik veriler
veriler.isnull().sum()

x = veriler.iloc[:,1:2].values
y = veriler.maas.values.reshape(-1,1)


sc_x = StandardScaler()
x_scale = sc_x.fit_transform(x)

sc_y = StandardScaler()
y_scale = sc_y.fit_transform(y)



#model
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scale, y_scale)


plt.scatter(x_scale, y_scale, color="blue", label = "Real")
plt.plot(x_scale, svr_reg.predict(x_scale),color = "red", label = "svr")
plt.legend()
plt.show()









