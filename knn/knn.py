from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("veriler.csv")
data.head()
data.info()
data.corr()
data.describe()
data.isnull().sum()
data.duplicated().sum()


x = data.iloc[:,1:4].values
y = data.iloc[:,-1].values.reshape(-1,1)

sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.3,
                                                    random_state = 42)

#model                    k komşudan bak   
knn = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
knn.score(x_test,y_test)
cm = confusion_matrix(y_test, y_pred)















