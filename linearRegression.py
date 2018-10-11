
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
print("########## Part 1 ##########")
hwdata = pd.read_csv(os.path.join(cwd, 'height_weight1.csv'))
hwdata.columns = ['height', 'weight']
hwdata.head()

### VARIABLE DESCRIPTIONS

# Height - in inches
# Weight - in lbs

hwdata.describe()

### Split data into training, test

hwdataTrain, hwdataTest = train_test_split(hwdata, test_size=.3, random_state=5203)
plt.plot(hwdataTrain[['height']], hwdataTrain[['weight']], ".")

### Creating Intercept Model
int_model = smf.ols(formula='weight ~ 1 + height', data=hwdataTrain).fit()
int_model.summary()

### Test our residuals
# Test for normalality of residuals
plt.hist(int_model.resid, 50)
plt.show()
# Residuals seem to be normal, with mean 0.

# Test for heteroscedasticity
plt.plot(int_model.predict(hwdataTrain), int_model.resid, '.')
# The prediction plot seems to show constant variance, with no heteroscedasticity.

### Creating No Intercept Model
no_int_model = smf.ols(formula='weight ~ 0 + height', data=hwdataTrain).fit()
no_int_model.summary()

### Test our residuals
# Test for normalality of residuals
plt.hist(no_int_model.resid, 50)
plt.show()

# Test for heteroscedasticity
plt.plot(no_int_model.predict(hwdataTrain), no_int_model.resid, '.')
# The prediction plot seems to show constant variance, meaning there is no heteroscedasticity.

#(1)TODO:have a stat guy look at
# The R-squared variance is less for our linear model with the intercept meaning that our data fits the first linear model better. The error variance is within a near range of y for each x as x and y increases, this is more true for our first model than our second so we can say this is a better fit. The errors are independent of each other we see this by the absence of any major defects in our graph along with no abnormal clumps of data in either graph.
#If we were to use R^2 as our function then our model then our residual would be at least even greater, so it wont fit the data better.


########## Part2 ##########
print("########## Part 2 ##########")

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

########## Part 3 ##########
print("########## Part 3 ##########")

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
