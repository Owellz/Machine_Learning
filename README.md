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

X = df[['Weight', 'Volume']] #independent
y = df['CO2'] #dependent

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

x = numpy.random.normal(3, 1, 100) #independent
y = numpy.random.normal(150, 40, 100) / x  #dependent

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



