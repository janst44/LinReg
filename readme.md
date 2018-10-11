### Big Data Science

# Linear Regression Assignment (Individual)

For all problems, submit your python code. Comment out your responses in the code
and submit 1 file.

For problems 1 and 2, the height_weight files contain 1 column of heights (in inches)
and 1 column of weights (in pounds). We are interested in the effect of height on
weight.

1. Using the height_weight1.csv file, fit a linear model to the height and weight
  data that includes an intercept. Fit another model that does not include the
  intercept. In terms of our 3 residual assumptions, how do these two models
  compare? Can we use R^2 or another metric to determine which fits the data
  better?
2. Using the heigh_weight2.csv data, fit another linear model. This time, do not
  include an intercept. Does this model meet the assumptions of residuals? If
  not, explain why not. Additionally, if one of the assumptions is not met, how
  do you think that impacts our prediction? Does it impact our point
  prediction? What about our prediction interval?

For Problem 3, the cars.csv file has age of the car (in years), the brand/make of a car
(Toyota, Ford, or BMW), the type (1 for SUV, 2 for sedan, and 3 for convertible),
number of miles, and price the car recently sold for.

3. Our goal is to determine a model to predict the price of a used car. Fit 3
  different models to the data and pick the “best” one. Why do you consider
  this model the best? Using this “best model”, predict what a 7 year old, BMW
  convertible with 67,000 miles would cost.


# Linear Regression Assignment 2 (Group)

This assignment should be done in your groups. Feel free to use Python or R for this assignment.

The data in icudata.csv are 200 observations from a much larger study on survival of patients admitted to the ICU.
1. Fit a logistic regression model that uses age (AGE), race (RACE), whether or not CPR was administered (CPR, 0 for no, 1 for had been administered), systolic blood pressure at admission (SYS), heart rate at admission (HRA), and type of admission (TYP). Do not do any parameter selection. Create a table with the coefficients from the regression.
2. Explain the effect CPR has on survival. How much does receiving CPR increase your likelihood to survive?
3. Now, using the same variables fit a model using the lasso technique. Using cross-validation, what is the optimal alpha value?
4. Using the alpha value and model from part (c), what are the coefficients for each of the parameters that were originally put in the model?

TED_talk_data.csv contains data for 2,550 TED talks. We are interested in finding a way to predict the number of views a talk will get.
1. Using the tag column, create a new column for each unique tag called TAG_xxx where xxx is the name of the tag in the tag column. The value will be TRUE or FALSE based on if the tag is contained in that talk’s tag column. For example, if the value for the tag column is [‘children’, ‘creativity’] in row 1, there will be a new column TAG_children with TRUE, TAG_creativity with TRUE, and all other TAG_xxx columns will be false.
2. Using the ratings column, create a new column for each rating category (14 in total). The value will be the count for the associated category for each row. For example, if the value is [{'id': 7, 'name': 'Funny', 'count': 19645}, {'id': 1, 'name': 'Beautiful', 'count': 4573}]. Then the RATINGS_Funny column will be 19645 and RATINGS_Beautiful column will be 4573.
3. Using LASSO, fit a model using comments, duration, number of speakers (num_speaker), the tag data (TAGS_xxx), and the ratings data (RATINGS_xxx). The TAG_ columns are true and false based on if the talk was tagged with the specified tag. The RATINGS_ columns the number of “votes” for the rating specified. **Note**: There are additional columns in the dataset, ignore these.
4. Using Cross-Validation, what is the optimal value for lambda?
5. What are the top 10 best and 10 worst topics (tags) for gaining viewership? What are the coefficients for these topics?
6. What is the least important characteristic (rating) of a TED talk in terms of getting more views?
