# ensemble learning
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor


#veri yÃ¼kleme
veriler =  pd.read_csv("maaslar.csv")

#veri Ã¶n iÅleme
#rastgele 10 ornek
veriler.sample(10)
#veriler hakkÄ±nda  bilgi ogrenme
veriler.info()
#sutunlara ulaÅma
veriler.columns
#eksik veriler
veriler.isnull().sum()

x = veriler.iloc[:,1:2].values
y = veriler.maas.values.reshape(-1,1)

#model
random_tree = RandomForestRegressor(n_estimators = 10, random_state=42)
random_tree.fit(x, y)


#visualize
plt.scatter(x, y, color="blue", label = "Real")
plt.plot(x, random_tree.predict(x),color = "red", label = "Random Forest")
plt.legend()
plt.show()
