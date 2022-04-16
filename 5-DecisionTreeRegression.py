# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:34 2020

@author: mfatihaltun
                
                Pv Generation Forecasting Decision Tree Regression
"""
                        # DATA PREPROCESSING #
## import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load to dataset
veriler = pd.read_csv("pvdata.csv")
veri1 = veriler.iloc[:, 6:13]
veri2 = veriler.iloc[:, 15:]

## convert categorical data into numerical data
dl = veriler.iloc[:,5:6].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dl[:,0] = le.fit_transform(veriler.iloc[:,5:6])
ohe = preprocessing.OneHotEncoder()
dl = ohe.fit_transform(dl).toarray()
dllast = pd.DataFrame(data=dl[:,1:], index=range(2920), columns=['Is Daylight'])

# Concat the DataFrames & independent variable ↓↓↓ (X)
veri1son = pd.concat([dllast,veri1], axis=1)
   
"""
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcek = sc1.fit_transform(veri1son)
sc2 = StandardScaler()
y_olcek = sc2.fit_transform(veri2)
"""
                        # MODEL BUILDING #
## build a decision tree regression model
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(veri1son,veri2)

y_tahmin = r_dt.predict(veri1son)
y_tahmin = y_tahmin.reshape(-1,1)
veri2 = veri2.values

#plt.scatter(X,Y, color='red')
plt.plot(y_tahmin[0:20,:], color = 'blue')
plt.plot(veri2[0:20,:], color = 'red')
plt.title("PV Generation Forecasting 2.Degree Polinomial")
plt.xlabel("Index")
plt.ylabel("PV Generation")
plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("\n","PV Generation Forecasting with 2.degree Polinomial Regression Method:", r2_score(veri2, y_tahmin),"\n")