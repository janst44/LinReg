 
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



#THE THREE CONDITIONS:
#1.The error variable is normally distributed.
#2.The error variance is constant for all values of x.
#3.The errors are independent of each other.
##########################Part1###################
#print(cwd)
print("\n\n#################PART1###############")
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

#plt.plot(hwdataTrain[['height']], hwdataTrain[['weight']], '.')
#plt.show()
model = smf.ols(formula='weight ~ 1 + height', data=hwdataTrain).fit()
print("Model with intercept summary")
print(model.summary())

### Test our residuals
print("Test of residuals:")
residual = model.resid
print(model.summary())

# Test for normal residuals
#plt.hist(residual, 50)
#plt.show()
# Test for heteroscedasticity
#plt.plot(model.predict(hwdataTrain), residual, '.')
#plt.show()

without_intercept = smf.ols(formula='weight ~ 0 + height', data=hwdataTrain).fit()
print("Model without intercept summary")
print(without_intercept.summary())

### Test our residuals
residual = without_intercept.resid
# Test for normal residuals
#plt.hist(residual, 50)
#plt.show()
# Test for heteroscedasticity
#plt.plot(without_intercept.predict(hwdataTrain), residual, '.')
#plt.show()


#(1)
#The error variance is normally distributed we see in our model without the intercept, this is we can see from the bell-curve histogram. The error variance is constant for all values as show from the plot of the points. No abnormal clumps of data in the graph w/without the intercept so the data is independent.
#If we were to use R^2 then that would help us acheive a higher R^2 to fit the data even better.

###################Part2###########################
print("\n\n############PART2###########")

hwdata = pd.read_csv(os.path.join(cwd, 'height_weight2.csv'))
hwdata.columns = ['height', 'weight']

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=123)

model = smf.ols(formula='weight ~ 0 + height', data=hwdataTrain).fit()
print("Model summary without intercept")
print(model.summary())

residual = model.resid
#plt.hist(residual, 50)
#plt.show()


#(2)
#THis model does meet all 3 residual assumptions. If one were not met however, that would mean that our predictions would have a larger margin for error. This is also true of point prediction, this is because the prediction interval will have increased and so point prediction is no longer accurate.

###################Part3###########################
print("\n\n############PART3###########")

cardata = pd.read_csv(os.path.join(cwd, 'car.csv'))
cardataTrain, cardataTest = train_test_split(cardata, test_size=.3, random_state=123)

print("\n\n#########  Model 1:   ###########")
model = smf.ols(formula='Price ~ 1 + Age +  Miles + C(Make) + C(Type)', data=cardataTrain).fit()
print(model.summary())

print("\n\n##########    Model 2:   ##########")
model = smf.ols(formula='np.log(Price) ~ 1 + np.log(Age) +  np.log(Miles) + C(Make) + C(Type)', data=cardataTrain).fit()
print(model.summary())

#Best fit model
print("\n\n###########    Model 3 and best fit:   ############")
model = smf.ols(formula='Price ~ 1 + np.log(Age) +  np.log(Miles) + C(Make) + C(Type)', data=cardataTrain).fit()
print(model.summary())

#residual = model.resid
#plt.hist(residual, 50)
#plt.show()


#(3) Our prediction of a 7 year old BMW with 67,000 miles: 25661.29, below is math using coefficients and our intercept using our model of best-fit:
#   5.78*10^4  +  -1776.7873 * 7 + -.1553 * 67000 + -9296.0945 * 1 = 25661.29

