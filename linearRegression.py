
### Logistic Regression

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

import os

cwd = os.getcwd()



# THE THREE CONDITIONS:
# 1. The error variable is normally distributed.
# 2. The error variance is constant for all values of x.
# 3. The errors are independent of each other.

########## Part1 ##########
print("########## Part1 ##########")
hwdata = pd.read_csv(os.path.join(cwd, 'height_weight1.csv'))
hwdata.columns = ['height', 'weight']
hwdata.head()

### VARIABLE DESCRIPTIONS

# Height - in inches
# Weight - in lbs

hwdata.dtypes
hwdata.describe()

### Split data into training, test

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=5203)
hwdataTrain.shape
hwdataTest.shape

# I don't think this is needed -A
# plt.plot(hwdataTrain[['height']], hwdataTrain[['weight']], hwdataTrain)

model = smf.ols(formula='weight ~ 1 + height', data=hwdataTrain).fit()
model.summary()

### Test our residuals
residual = model.resid

# Test for normalality of residuals
plt.hist(residual, 50)
plt.show()

# Test for heteroscedasticity
plt.plot(model.predict(hwdataTrain), residual, '.')

model_rsquared = smf.ols(formula='weight ~ 0 + height', data=hwdataTrain).fit()
model_rsquared.summary()

### Test our residuals
residual = model_rsquared.resid

# Test for normalality of residuals
plt.hist(residual, 50)
plt.show()

# Test for heteroscedasticity
#plt.plot(model_rsquared.predict(hwdataTrain), residual, '.')


#(1)TODO:have a stat guy look at
#The R-squared variance is less for our linear model with the intercept meaning that our data fits the first linear model better. The error variance is within a near range of y for each x as x and y increases, this is more true for our first model than our second so we can say this is a better fit. The errors are independent of each other we see this by the absence of any major defects in our graph along with no abnormal clumps of data in either graph.
#If we were to use R^2 as our function then our model then our residual would be at least even greater, so it wont fit the data better.

###################Part2###########################
print("############PART2###########")

hwdata = pd.read_csv(os.path.join(cwd, 'height_weight2.csv'))
hwdata.columns = ['height', 'weight']

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=123)
hwdataTrain.shape
hwdataTest.shape

model = smf.ols(formula='weight ~ 0 + height', data=hwdataTrain).fit()
print(model.summary())

residual = model.resid
plt.hist(residual, 50)
plt.show()


#(2)TODO:
#

###################Part3###########################
print("############PART3###########")

hwdata = pd.read_csv(os.path.join(cwd, 'car.csv'))
hwdata.columns = ['age', 'make', 'type', 'miles', 'price']

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=123)
hwdataTrain.shape
hwdataTest.shape

model = smf.ols(formula='price ~ age + type + miles', data=hwdataTrain).fit()
print(model.summary())

residual = model.resid
plt.hist(residual, 50)
plt.show()


#(3)TODO:
#
