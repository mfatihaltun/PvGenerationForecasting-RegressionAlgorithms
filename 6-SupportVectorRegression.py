# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:34 2020

@author: mfatihaltun

            Pv Generation Forecasting Support Vector Machine Regression
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
# numpy to Dataframe
dllast = pd.DataFrame(data=dl[:,1:], index=range(2920), columns=['Is Daylight'])

# Concat the DataFrames
veri1son = pd.concat([dllast,veri1], axis=1)
   
## Standardize the variables  
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcek = sc1.fit_transform(veri1son)
sc2 = StandardScaler()
y_olcek = sc2.fit_transform(veri2)

                        # MODEL BUILDING #
## build a Support vector machine regression model
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcek, y_olcek)
y_tahmin = svr_reg.predict(x_olcek)
y_tahmin = y_tahmin.reshape(-1,1)

# to show R2 score
from sklearn.metrics import r2_score
print("\n", "PV Generation Forecasting with SVR Method:",r2_score(y_olcek, y_tahmin),"\n")

# to show graph
plt.plot(y_tahmin[0:20,:], color = 'blue')
plt.plot(y_olcek[0:20,:], color = 'red')
plt.title("PV Generation Forecasting SVR")
plt.xlabel("Index")
plt.ylabel("PV Generation")
plt.show()
