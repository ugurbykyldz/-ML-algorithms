from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder , OneHotEncoder ,StandardScaler
from sklearn.compose import ColumnTransformer 
from xgboost import XGBClassifier


import numpy as np
import pandas as pd


data = pd.read_csv("Churn_Modelling.csv")
data.sample(10)
data.columns
data.duplicated().sum()
data.isnull().sum()
data.info()
data.corr()
data.describe()


x = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values.reshape(-1,1)

#one hot encoding
label_encoding = LabelEncoder()
ohe = OneHotEncoder()

x[:, 1] = label_encoding.fit_transform(x[:, 1])
#dummy variable dikkat
x[:,2] = label_encoding.fit_transform(x[:, 2])

ulke = x[:,1].reshape(-1,1)
ulke = ohe.fit_transform(ulke).toarray()

ilk_sutun = x[:,0].reshape(-1,1) 
x = np.concatenate([ilk_sutun ,ulke,  x[:,2:]] , 1)

#scale
x = StandardScaler().fit_transform(x)


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state = 42)


#model
xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test )

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(xgb.score(x_test, y_test))


