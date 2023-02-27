

"""
DLMDSPWP01_Written Assessment

Given: 
    (A) 4 training datasets
    (B) One test dataset
    (C) Datasets for 50 ideals
    
    X     Y
    X1    Y1
    .     .
    .     .
    Xn    Yn
    
Uses training data to CHOOSE the four ideal functions which are the best fit out of the fifty provided

"""

"""
Modeling building

Define: What type of model will be? A decision tree? Some type of model? Some other parameters of the model
type are specified too.
Fit: Capture patterns from provide. This is heart of modelling
Predict: Just what it sounds like
Evaluate: Determine how accurate the model predictions are

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train_file = 'train.csv'
test_file = 'test.csv'
ideal_file = 'ideal.csv'

df = pd.read_csv(train_file)

#Set X & Y values
x = df.x
y = [df.y1, df.y2, df.y3, df.y4]
n = df.shape[0]

#Save the slope, y-intercept, y-deivation squared

slope_list = []

y_intercept_list = []

ysd_list = []

#My Line & My Model
myline = np.linspace(-20, 22, 400)

y_value = 0

 
def least_Sq(x, y, n):
    
    global y_value
    
    # Sigma Variables
    Sig_X = sum(x)
    Sig_Y = sum(y)
    Sig_XY = sum(x*y)
    Sig_X_sq = sum(x*x)
    
    
    #Formula of the Slope
    m = (n*(Sig_XY) - (Sig_X * Sig_Y)) / ((n*Sig_X_sq) - (Sig_X * Sig_X))
    
    #Save slope
    slope_list.append(m)
    
    print("The Slope is: " + str(m))
    
    # Formula of the constant (Y-intercept)
    b = (Sig_Y - (m*Sig_X)) / n
    
    # Save y-intercept
    
    y_intercept_list.append(b)
    
    print("The Y-intercept is: " + str(b))
    
    #Least Squared Formula
    
    y_value = m*x + b
    
def plot(x, y):
    # Plot for checking
    plt.plot(x, y, label="Data", linewidth=2, color='red')
    
    mymodel = np.poly1d(np.polyfit(x, y_value, 4))
    
    plt.plot(myline, mymodel(myline), label="Least Squared Line", linewidth=2, color='blue')

    plt.legend()
    plt.grid(True, color="k")
    plt.ylabel('x')
    plt.xlabel('y')
    plt.show()
    

def ysd(y, n):
    
    # Find the mean of y
    
    y_mean = sum(y) / n
    
    #Subtract each value with mean
    
    y_sub = y - y_mean
    
    #Squared the new value
    
    y_sq = y_sub * y_sub
    
    #Sum the squared
    
    ysd = sum(y_sq)
    
    ysd_list.append(ysd)
    

def Train_Data():
    
    #Find the least squared function for each the training data function (y)
    for i in y:
        #Calculate Least Squared Equation
        least_Sq(x, i, n)
        
        print(y_value)
        
        # Test Ploting Data
        plot(x, i)
        
        # Sum of y-squared(varience) deviation
        ysd(i,n)
        
        print(ysd_list)


    
if __name__ == '__main__':
    Train_Data()