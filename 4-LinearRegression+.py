# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:34 2020

@author: mfatihaltun

            Pv Generation Forecasting Linear Regression with scaling 
                daylight loop added. 
"""
                        # DATA PREPROCESSING #
## import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load to dataset
veriler = pd.read_csv("pvdata.csv")
###print(veriler, '\n')
###veri0 = veriler.iloc[:,4:5]
veri1 = veriler.iloc[:, 6:15]
veri2 = veriler.iloc[:, 15:]
### ↑↑↑ veri2 : dependent variable (Y)
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

# Concat the DataFrames & independent variable ↓↓↓ (X)
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


                        # MODEL BUILDING #
## build a linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_tahmin = lin_reg.predict(x_test) 

## day light 0 or 1
is_daylight_list = list(x_test["Is Daylight"])
for counter in range(len(x_test)):
    if is_daylight_list[counter] == 0.0:
        y_tahmin[counter] = 0.0

# show to graph
plt.plot(y_tahmin[0:10,:], color = 'blue')
plt.plot(y_test.iloc[0:10,:], color = 'red')
plt.show()

# to show R2 score, mserror, maerror
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print(r2_score(y_test, y_tahmin))
#print('-----------------------')
#print(y_test, '\n')
print(mean_squared_error(y_test, y_tahmin, squared=False))
print(mean_absolute_error(y_test, y_tahmin))


### OLS Report 
import statsmodels.api as sm
X = np.append(arr = np.ones((2920,1)).astype(int), values=veri3, axis=1)
X_l = veri4.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(veri2, X_l).fit()
print(model.summary())
print('********************-------------**********************')


