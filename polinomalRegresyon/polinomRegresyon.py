# y = b0 + b1x^1 + bnx^n  + E 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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

#simple linear
linear_regresyon = LinearRegression()
linear_regresyon.fit(x, y)

#visualize
plt.scatter(x, y, color = "blue", label = "Real")
plt.plot(x, linear_regresyon.predict(x), color = "red", label = "Linear Regresyon")
plt.legend()
plt.show()


#polynomial regresyon
poly_regresyon = PolynomialFeatures(degree = 2)
#x çevir x^0 + x^1 + x^2
x_poly = poly_regresyon.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


#visualize
plt.scatter(x, y, color = "blue", label = "Real")
plt.plot(x, linear_regresyon.predict(x), color = "red", label = "Linear Regresyon")
plt.plot(x, lin_reg2.predict(poly_regresyon.fit_transform(x)), color = "green", label = "Polynomial Regresyon")
plt.legend()
plt.show()















