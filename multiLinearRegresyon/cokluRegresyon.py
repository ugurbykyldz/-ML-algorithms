# y = b0 + b1X1  + b2X2 + bnXn + E
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#veri yükleme
veriler =  pd.read_csv("veriler.csv")

#veri ön işleme
#rastgele 10 ornek
veriler.sample(10)
#veriler hakkında  bilgi ogrenme
veriler.info()
#sutunlara ulaşma
veriler.columns
#eksik veriler
veriler.isnull().sum()

#numericleri alma
yasBoyKilo = veriler.iloc[:,1:-1]



#katagorik veriler nominal(10011100)  
#ülke için
ulke = veriler.iloc[:,0:1].values

labelEncod = preprocessing.LabelEncoder()

ulke[:,0] = labelEncod.fit_transform(veriler.iloc[:,0])


#one hot encoding nominal 0011010 gibi
onehotencoding = preprocessing.OneHotEncoder()
ulke = onehotencoding.fit_transform(ulke).toarray()


#cinsiyet
cinsiyet = veriler.iloc[:,-1].values.reshape(-1,1)

labelEncod = preprocessing.LabelEncoder()

cinsiyet[:,0] = labelEncod.fit_transform(cinsiyet[:,0])

#one hot encoding nominal 0011010 gibi
onehotencoding = preprocessing.OneHotEncoder()
cinsiyet = onehotencoding.fit_transform(cinsiyet).toarray()


#birleştirme
sonucUlke = pd.DataFrame(data = ulke ,index = range(ulke.shape[0]), columns = ["fr","tr","us"])

sonucCinsiyet = pd.DataFrame(data = cinsiyet, index = range(cinsiyet.shape[0]), columns = ["E","K"])


data = pd.concat([sonucUlke, yasBoyKilo, sonucCinsiyet] , axis = 1)

#dummy variable tuzagı dikkat  1 sutun 2 degişkeni ifade ediyor diger sutunu sil
data.drop(["K"] ,inplace = True, axis = 1)
data = data.rename(columns = {"E":"cinsiyet"})



x = data.iloc[:,:-1]
y = data.iloc[:,-1]
#train test spilit

x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2, random_state=42)


#öznitelik ölçeklendirme
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

y_train =  y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)


#model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


#boy tahmini
x_boy = data.drop(["boy"], axis = 1).values
y_boy = data["boy"].values.reshape(-1,1)


x_train, x_test, y_train, y_test = train_test_split(x_boy ,y_boy  ,test_size=0.2, random_state=42)
lr_boy = LinearRegression()
lr_boy.fit(x_train, y_train)
y_pred_boy = lr_boy.predict(x_test)



#modeliin ve degişkenlerin başarısı -backward elimination-

import  statsmodels.api as sm
X = np.append(arr = np.ones((x_boy.shape[0],1)).astype(int), values = x_boy, axis = 1) # b0 = 1 ekliyoz formulden 

x_1 = x_boy[:, [0,1,2,3,4,5]]
x_1 = np.array(x_1 , dtype=float)
model = sm.OLS(y_boy, x_1).fit()
print(model.summary()) # p = 0.05 ele degişken çürütme

#x5 ele yani 5. index   4 sil
x_1 = x_boy[:, [0,1,2,3,5]]
x_1 = np.array(x_1 , dtype=float)
model = sm.OLS(y_boy, x_1).fit()
print(model.summary())













