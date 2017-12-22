# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 23:29:40 2017

@author: smit
"""

#simple regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as ple

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=1/3,random_state=0)

#implementing simple linear regression
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(x_train1,y_train1)
y_pred=linear.predict(x_test1)

#visualizing data
ple.scatter(x_train1,y_train1,color='red')
ple.plot(x_train1,linear.predict(x_train1),color='blue')
ple.title('salary vs experience training data')
ple.xlabel('years of experience')
ple.ylabel('salary')
ple.show()

#test data
ple.scatter(x_test1,y_test1,color='red')
ple.plot(x_train1,linear.predict(x_train1),color='blue')
ple.title('salary vs experience testing data')
ple.xlabel('years of experience')
ple.ylabel('salary')
ple.show()