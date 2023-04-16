# Machine_Learning



# Using Linear regression of two values to determine the Relationship (r) between them
#-1, 1 means their is relationship while 
# 0 meansno relationship

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)

## Execute a method that returns some important key values of Linear Regression:

slope, intercept, r, p, std_err = stats.linregress(x, y)



## To Predict the speed of a 10 years old car:

from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

speed = myfunc(10)

print(speed)



# Using SCALE to predict Predict of CO2 emission from a 1.3 liter car that weighs 2300 kilograms:

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)


## using Train and Test Model to create the relationship between 100 customers and thier 
## Spending habit - 
--- 80% for training, and 20% for testing was used to evaluate the process.

import numpy
import matplotlib.pyplot as plt

numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[20:]
test_y = y[20:]

plt.scatter(train_x, train_y)
plt.show()

Result:
The x axis represents the number of minutes before making a purchase.

The y axis represents the amount of money spent on the purchase.


## Draw a polynomial regression line through the data points:

import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()


## Making a prediction of  


import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library
music_data = pd.read_csv('music.csv')
music_data

## split data by dropping the colunms 

X = music_data.drop(columns=['genre'])
X

## Split to Output data ( is the prediction for the answer , training to make predictions for the model)

y = music_data['genre']
y

#  Build a Model; Using Decision Tree as the Machine algorithm ( from Sklearn library)

from sklearn.tree import DecisionTreeClassifier

## create an object and set to an instance - the model
model = DecisionTreeClassifier()
## train it to learn patterns in the data
model.fit(X, y)
## now make prediction of a 21 year old male and 22 years female music they likes
predictions = model.predict([[21, 1], [22, 0]])
predictions ## and click enter on the variable

# Measure accuracy of Model ( splitting the model into Training and Testing )

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## create an object and set to an instance - the model
model = DecisionTreeClassifier()
#allocate 80% for training and 20% for testing to calculate the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## X_train, X_test are the input set while 
## y_train, and y_test are the output set.

model.fit(X_train, y_train) ## passing only the train data set to evaluate the model
predictions = model.predict(X_test) ## X_test contains the input values for testing

## To get the accurcy, add a library for this
## and compare prediction by the actual values we have for testing

score = accuracy_score(y_test, predictions) 
## y_test is the expected values and prediction contains actual values
score

# Model Persistence ( no need to train a model if you have a new users, only by loading already trained model from a file )
#To save a trained model for future model testing


import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library

## as metohd for saving and loading model use this for model persistency
import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

## create an object and set to an instance - the model
model = DecisionTreeClassifier()
## train it to learn patterns in the data
model.fit(X, y)


joblib.dump(model, 'music-recommender.joblib') ## save and laoding


## now make prediction of a 21 year old male and 22 years female music they likes
#predictions = model.predict([[21, 1], [22, 0]])
#predictions ## and click enter on the variable

## Model Persistence ( TO LOAD AN EXISTING MODEL )

import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library
## as metohd for saving and loading model use this for model persistency
import joblib
## laoding an exisiting model an make prediction with it

model = joblib.load('music-recommender.joblib') 

## now make prediction of a 21 year old male and 22 years female music they likes
predictions = model.predict([[21, 1], [22, 0]])
predictions ## and click enter on the variable

# VISUALISING THE MODEL PREDICTION

import pandas as pd
from sklearn.tree import DecisionTreeClassifier ## library
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

## create an object and set to an instance - the model
model = DecisionTreeClassifier()

model.fit(X, y)

tree.export_graphviz(model, out_file = 'music-recommender.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
