# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:42:44 2022

@author: hashem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

plt.scatter(X, y, color="red")
plt.plot(X, lr.predict(X), color="blue")
plt.title("linear regression")
plt.xlabel("level")
plt.ylabel("Salary")
plt.show()



from sklearn.preprocessing import PolynomialFeatures
pf= PolynomialFeatures(degree=10)
X_poly=pf.fit_transform(X)

lr2 = LinearRegression()
lr2.fit(X_poly, y)

plt.scatter(X, y, color="red")
plt.plot(X, lr2.predict(X_poly), color="blue")

