# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 00:06:47 2017

@author: smit
"""
#data preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,3]

#missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
x.iloc[:,1:3]=imputer.fit_transform(x.iloc[:,1:3])

#catagorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x.iloc[:,0]=labelencoder.fit_transform(x.iloc[:,0])
onehot=OneHotEncoder(categorical_features=[0])
x=onehot.fit_transform(x).toarray()
labelencoder1=LabelEncoder()
y=labelencoder1.fit_transform(y)

#splitting data in training and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

