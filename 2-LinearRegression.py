# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:34 2020

@author: mfatihaltun

        Pv Generation Forecasting Linear Regression with scaling 
"""
                         # DATA PREPROCESSING #
## import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load to dataset
veriler = pd.read_csv("pvdata.csv")
#print(veriler, '\n')

veri1 = veriler.iloc[:, 6:15]
veri2 = veriler.iloc[:, 15:]
veri1sag = veri1.iloc[:,7:].values
veri1sol = veri1.iloc[:,0:7]

## missing values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(veri1sag[:,:])
veri1sag[:,:] = imputer.transform(veri1sag[:,:])
# numpy to Dataframe
veri1sag = pd.DataFrame(data=veri1sag[:,0:], index=range(2920), columns=['Average Wind Speed (Period)','Average Barometric Pressure (Period)'])

#print(veri1sag.isnull().sum())
# to show number of total null values

## convert categorical data into numerical data
dl = veriler.iloc[:,5:6].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dl[:,0] = le.fit_transform(veriler.iloc[:,5:6])

ohe = preprocessing.OneHotEncoder()
dl = ohe.fit_transform(dl).toarray()
# numpy to Dataframe
daylight = pd.DataFrame(data=dl[:,1:], index=range(2920), columns=['Is Daylight'])

# Concat the DataFrames
veri3 = pd.concat([veri1sol,veri1sag], axis=1)
veri4 = pd.concat([daylight, veri3], axis=1)

## Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(veri4, veri2, test_size=0.33, random_state=0)

## sort the variables
x_train = x_train.sort_index()
x_test = x_test.sort_index()
y_train = y_train.sort_index()
y_test = y_test.sort_index()

## Standardize the variables  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


                        # MODEL BUILDING #
## build a linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
y_tahmin = lin_reg.predict(X_test) 

# to show graph
##print(y_test.iloc[0:20,:] ,'\n', y_tahmin[0:20,:])
plt.plot(y_tahmin[0:20,:], color = 'blue' ,label='tahmin')
plt.plot(Y_test[0:20,:], color = 'red' ,label='gerçek')
plt.show()

# to show R2 score, mserror, maerror
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print(r2_score(Y_test, y_tahmin), '\n')
print(mean_squared_error(Y_test, y_tahmin, squared=False), '\n')
print(mean_absolute_error(Y_test, y_tahmin), '\n')

"""
### OLS REPORT
import statsmodels.api as sm
X = np.append(arr = np.ones((2920,1)).astype(int), values=veri4, axis=1)
X_l = veri4.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(veri2, X_l).fit()
print(model.summary(), '\n')
"""
