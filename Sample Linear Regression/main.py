# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 16:29:48 2023

@author: Sophanith
"""

# Code source: Jaques Grobler
# License: BSD 3 clause
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#Load training data
training_list = ['Training_01.csv', 'Training_02.csv', 'Training_03.csv', 'Training_04.csv']

df_0 = pd.read_csv(training_list[0], nrows=50).to_numpy()
df_1 = pd.read_csv(training_list[1], nrows=50).to_numpy()
df_2 = pd.read_csv(training_list[2], nrows=50).to_numpy()
df_3 = pd.read_csv(training_list[3], nrows=50).to_numpy()


X_train_0 = np.array(df_0[:,0]).reshape(-1,df_0.shape[0])
y_train_0 = np.array(df_0[:,1]).reshape(-1,df_0.shape[0])


#Set X & Y training values

"""
#Set X training values
X_train_0 = df_0.Height
X_train_1 = df_1.Height
X_train_2 = df_2.Height
X_train_3 = df_3.Height

#Set Y training values
X_train_0 = df_1.Height
y_train_1 = df_1.Weight
y_train_2 = df_2.Weight
y_train_3 = df_3.Weight
"""
#Set X & Y test values
df_test = pd.read_csv('Test_01.csv').to_numpy()
X_test = np.array(df_test[:,0]).reshape(-1,df_test.shape[0])
y_test = np.array(df_test[:,1]).reshape(-1,df_test.shape[0])

#Set X & Y train values
df_prod = pd.read_csv('Production-01.csv', nrows=50).to_numpy()
X_prod = np.array(df_prod[:,0]).reshape(-1,df_prod.shape[0])
y_prod = np.array(df_prod[:,1]).reshape(-1,df_prod.shape[0])


"""
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
"""

# Code here
# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
filename = 'finalized_model.sav'

"""for i in training_list:
    df_0 = pd.read_csv(i, nrows=50).to_numpy()
"""

X_train_0 = np.array(df_0[:,0]).reshape(-1,df_0.shape[0])
y_train_0 = np.array(df_0[:,1]).reshape(-1,df_0.shape[0])
regr = joblib.load(filename)
regr.fit(X_train_0, y_train_0)
joblib.dump(regr, filename)

"""
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
"""

# Make prediction using the testing set
print('X_test:')
print(X_test)
y_pred = regr.predict(X_test)

print('y_pred:')

# Ends here

"""

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
"""


#The coefficients
print("Coefficients: \n", regr.coef_)
#The mean squared error
print("Mean squared error:  %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Plot outputs
plt.scatter(X_test, y_pred, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

