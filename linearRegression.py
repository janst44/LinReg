 
# coding: utf-8

# # Logistic Regression

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

import os

cwd = os.getcwd()

##########################Part1###################
#print(cwd)
print("PART1")
hwdata = pd.read_csv(os.path.join(cwd, 'height_weight1.csv'))
hwdata.columns = ['height', 'weight']
#print(hwdata.head())

# ##### VARIABLE DESCRIPTIONS
#
# Height - in inches
# Weight - in lbs

#print(hwdata.dtypes)


#print(hwdata.describe())

### split data into training, test
hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=123)
hwdataTrain.shape
hwdataTest.shape

#plt.plot(hwdataTrain[['height']], hwdataTrain[['weight']], hwdataTrain)

model = smf.ols(formula='weight ~ 1 + height', data=hwdataTrain).fit()
print(model.summary())

### Test our residuals
residual = model.resid
# Test for normal residuals
plt.hist(residual, 50)
plt.show()
#print(plt.show())
# Test for heteroscedasticity
#plt.plot(model.predict(hwdataTrain), residual, '.')

model_rsquared = smf.ols(formula='weight ~ height -1', data=hwdataTrain).fit()
print(model_rsquared.summary())

### Test our residuals
residual = model_rsquared.resid
# Test for normal residuals
plt.hist(residual, 75)
plt.show()
# Test for heteroscedasticity
#plt.plot(model_rsquared.predict(hwdataTrain), residual, '.')

###################Part2###########################
print("PART2")

hwdata = pd.read_csv(os.path.join(cwd, 'height_weight2.csv'))
hwdata.columns = ['height', 'weight']

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=123)
hwdataTrain.shape
hwdataTest.shape

model = smf.ols(formula='weight ~ 1 + height', data=hwdataTrain).fit()
print(model.summary())

residual = model.resid
plt.hist(residual, 50)
plt.show()

model_rsquared = smf.ols(formula='weight ~ height -1', data=hwdataTrain).fit()
print(model_rsquared.summary())

residual = model_rsquared.resid
plt.hist(residual, 75)
plt.show()


#(1)
#The R-squared variance is less for our linear model with the intercept meaning that our data fits the first linear model better. The error variance is within a near range of y for each x as x and y increases, this is more true for our first model than our second so we can say this is a better fit. The errors are independent of each other we see this by the absence of any major defects in our graph along with no abnormal clumps of data in either graph.
#If we were to use R^2 as our function then our model then our residual would be at least even greater, so it wont fit the data better.
#(2)
#

