from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("musteriler.csv")
data.head(10)
data.info()
data.isnull().sum()
data.duplicated().sum()

x = data.iloc[:, 3:].values

kmeans = KMeans(n_clusters=3 , init = "k-means++")
kmeans.fit(x)

#merkez noktaları
kmeans.cluster_centers_


#WCSS en iyi cluster bulme best k
result = list()
for i in range(1,11):
    kmeans = KMeans(n_clusters =i, init = "k-means++", random_state=42)
    kmeans.fit(x)
    #wcss
    result.append(kmeans.inertia_)
    
    
    
#visualize dirsek noktası 4 olabilir
plt.plot(range(1,11), result, color = "red", marker = "+",mec = 'blue')
plt.xlabel("cluster(k)")
plt.ylabel("WCSS")
plt.show()
















    
    