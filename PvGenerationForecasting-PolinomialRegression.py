# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:34 2020

@author: mfatihaltun

                Pv Generation Forecasting with Polinomial Regression Model
"""
                            # DATA PREPROCESSING #
## import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load to dataset
veriler = pd.read_csv("pvdata.csv")
veri1 = veriler.iloc[:, 6:13]       # independent variable
veri2 = veriler.iloc[:, 15:]        # dependent variable


## convert categorical data into numerical data
dl = veriler.iloc[:,5:6].values
from sklearn import preprocessing
# label encoding
le = preprocessing.LabelEncoder()
dl[:,0] = le.fit_transform(veriler.iloc[:,5:6])
# one hot encoding
ohe = preprocessing.OneHotEncoder()
dl = ohe.fit_transform(dl).toarray()
# numpy array to DataFrame 
dllast = pd.DataFrame(data=dl[:,1:], index=range(2920), columns=['Is Daylight'])
veri1son = pd.concat([dllast,veri1], axis=1)        # last independent variable
   
## Standardize the variables  
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcek = sc1.fit_transform(veri1son)
sc2 = StandardScaler()
y_olcek = sc2.fit_transform(veri2)


                            # MODEL BUILDING #
## build a 2nd order polynomial regression model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)         
x_poly2 = poly_reg.fit_transform(x_olcek)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y_olcek)
y_tahmin2 = lin_reg2.predict(x_poly2)

## show to R2 score
from sklearn.metrics import r2_score
print("2.derece polinom tahmin:", r2_score(y_olcek, y_tahmin2),"\n")

## plot the graph
plt.plot(y_tahmin2[0:20,:], color = 'blue')
plt.plot(y_olcek[0:20,:], color = 'red')
plt.title("PV Generation Forecasting 2.Degree Polinomial")
plt.xlabel('Index')
plt.ylabel('PV Generation')
plt.show()


## build a 4th order polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly4 = poly_reg.fit_transform(x_olcek)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4, y_olcek)
y_tahmin4 = lin_reg4.predict(x_poly4)

from sklearn.metrics import r2_score
print("4.derece polinom tahmin:", r2_score(y_olcek, y_tahmin4),"\n")

plt.plot(y_tahmin4[0:20,:], color = 'blue')
plt.plot(y_olcek[0:20,:], color = 'red')
plt.title("PV Generation Forecasting 4.Degree Polinomial")
plt.xlabel('Time')
plt.ylabel('PV Generation')
plt.show()

