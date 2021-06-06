# kara agaçı
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeRegressor


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
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(x, y)


#visualize
plt.scatter(x, y, color="blue", label = "Real")
plt.plot(x, decision_tree.predict(x),color = "red", label = "Decision Tree")
plt.legend()
plt.show()


