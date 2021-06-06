from sklearn.cluster import AgglomerativeClustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("musteriler.csv")
data.head(10)
data.info()
data.isnull().sum()
data.duplicated().sum()

x = data.iloc[:, 3:].values
 
#hc
hc = AgglomerativeClustering(n_clusters=3, affinity = "euclidean", linkage="ward")
y_pred = hc.fit_predict(x)

    
    
#visualize 
plt.scatter(x[y_pred == 0 ,0] , x[y_pred ==0 ,1], s = 100 , c="red",label = "0")
plt.scatter(x[y_pred == 1 ,0] , x[y_pred ==1 ,1], s = 100 , c="blue",label = "1")
plt.scatter(x[y_pred == 2 ,0] , x[y_pred ==2 ,1], s = 100 , c="green",label = "2")
plt.legend()
plt.show()


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.show()




























    
    