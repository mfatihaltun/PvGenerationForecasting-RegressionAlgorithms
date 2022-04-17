# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:55:34 2020

@author: mfa

                >>> DATA PREPROCESSING AND LINEAR REGRESSION ALGORTIHM TEMPLATE <<<       
"""
                            # DATA PREPROCESSING #
## import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load to dataset and specify to dependent and independent variables 
veriler = pd.read_csv("veriler.csv")
boy = veriler[['boy']]    
yas = veriler.iloc[:,3:4]
### ↑↑↑ data frame variable
kilo = veriler.iloc[:,2:3].values
y = yas.values
### ↑↑↑ numpy array variable


## complete the missing values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
yas = veriler.iloc[:,3:4].values
### ↑↑↑ index location of the missing variables
### ↓↓↓ fit and transform 
imputer = imputer.fit(yas[:,3:4])
yas[:,3:4] = imputer.transform(yas[:,3:4])
### in first, to see the number of total missing value
print(veriler.isnull().sum())


## convert categorical data into numerical data
ulke = veriler.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
### label encoding  ↑↑↑
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
### one hot encoding ↑↑↑


### the shortcut of label encoding for all variables  ↓↓↓
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)


### converting numpy to data frame 
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
sonuc1 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])


### Concat the DataFrames
s = pd.concat([sonuc,sonuc1], axis=1)
s2 = pd.concat([X, Y], axis=1)


### Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)


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
lr = LinearRegression()
lr.fit(x_train, y_train)
### predict variable ↓↓↓
y_tahmin = lr.predict(x_test)


## sort the variables
x_train = x_train.sort_index()
y_train = y_train.sort_index()
x_test = x_test.sort_index()
y_test = y_test.sort_index()


### to show graphs
plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)
plt.title('Tahmin Grafiği')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
### 
plt.scatter(X,Y, color = 'red')
plt.plot(x, lin_reg.predict(X), color = 'blue')
plt.title('Tahmin Grafiği')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')


### to show results
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


### OLS report && BACKWARD ELIMINATION
import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values=s22, axis=1)
X_l = s22.iloc[:, [0,1,2,3,4,5]].values
###
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy1, X_l).fit()
print(model.summary())


## eliminate the variables with P value greater than 0.005   
X_l = s22.iloc[:, [0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy1, X_l).fit()

### prediction after backward elimination   
sol = x_train.iloc[:,0:3]
sag = x_train.iloc[:,4:]    
x_train = pd.concat([sol, sag], axis=1)

sol1 = x_test.iloc[:,0:3]
sag1 = x_test.iloc[:,4:]    
x_test = pd.concat([sol1, sag1], axis=1)

regressor.fit(x_train, y_train)
y_tahmin = regressor.predict(x_test) 


