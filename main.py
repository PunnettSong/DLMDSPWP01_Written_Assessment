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
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

def Train_Data():
    
    training_list = ['Training_01.csv', 'Training_02.csv', 'Training_03.csv', 'Training_04.csv']
    for i in training_list:
        df = pd.read_csv(i)
        
        #Set X & Y values
        x = df.Height
        y = df.Weight
        
        
        #Set least squared variables
        Sig_X = sum(x)
        Sig_Y = sum(y)
        Sig_XY = sum(x*y)
        Sig_X_sq = sum(x*x)
        n = df.shape[0]
        
        #Formula of the Slope
        m = (n*(Sig_XY) - (Sig_X * Sig_Y)) / ((n*Sig_X_sq) - (Sig_X * Sig_X))
        print("The Slope is: " + str(m))
        
        # Formula of the constant (Y-intercept)
        b = (Sig_Y - (m*Sig_X)) / n
        print("The Y-intercept is: " + str(b))
        
        #
        y_value = m*x + b
        
        print(x, y_value)
        
        
        
        #Plot data
        plt.scatter(x, y, label="Training_01")
        plt.plot(x, y_value, label="Line Plot", linewidth=2)
        plt.legend()
        plt.grid(True, color="k")
        plt.ylabel('Height')
        plt.xlabel('Weight')
        plt.show()
    
if __name__ == '__main__':
    Train_Data()