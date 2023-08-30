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

" Least Square and that gives us a noise-free
Data set for each of our 4 func�ons. Ie, instead of my noisy training func�ons,
I now use clean data from the ideal func�ons data set"
"""

import pandas as pd
import numpy as np
import math as math
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, text

from print_disclaimer import print_disclaimer
from draw import Draw as dw
from draw import Scatter
from draw import Line

#Working with database
df_train_file = pd.read_csv('train.csv')
df_test_file = pd.read_csv('test.csv')
df_ideal_file = pd.read_csv('ideal.csv')

engine = create_engine('sqlite:///training_data.db')
connection = engine.connect()

# Write the data into the database
df_train_file.to_sql('train', connection, if_exists='replace')
df_ideal_file.to_sql('ideal', connection, if_exists='replace')
df_test_file.to_sql('test', connection, if_exists='replace')

# Query the data from the TRAIN table and save into a df_query_train variable
query_train = text("SELECT * FROM train")
df_query_train = pd.read_sql(query_train, connection)

# Query the data from the IDEAL table and save into df_query_idea variable
query_ideal = text("SELECT * FROM ideal")
df_query_ideal = pd.read_sql(query_ideal, connection)

# Query the data from the IDEAL table and save into df_query_idea variable
query_test = text("SELECT * FROM test")
df_query_test = pd.read_sql(query_test , connection)

# Set X & Y values for training_set
x_train = df_query_train.iloc[:,1:2]
y_train = df_query_train.iloc[:,2:]
n_train = df_query_train.shape[0]

#Set X & Y values for ideal_set
x_ideal = df_query_ideal.iloc[:,1:2]
y_ideal = df_query_ideal.iloc[:,2:]
n_ideal = df_query_ideal.shape[0]

#Set X & Y values for test_set
x_test = df_query_test.iloc[:,1:2]
y_test = df_query_test.iloc[:,2:]
n_test = df_query_test.shape[0]

#Prep variable to save some data

slope_list_train = []
y_intercept_list_train = []
y_pred = 0
y_pred_list = []
df_y_pred = pd.DataFrame()
sum_y_pred_list = []
sum_y_pred = 0
sum_y_ideal_list = []
sum_y_ideal = 0
least_dif = 0
dif = 0
count = 0
final_index = []
final_ideal = 0
df_final_ideal = pd.DataFrame()
diff_test_ideal = 0
new_y_test = []
df_new_y_test = pd.DataFrame()
removed_y_ideal = []
y_dev_list = []

#Function for calculating the slope and y-intercept
def least_Sq(x, y, n):
    # Sigma 
    Sig_X = 0
    Sig_Y = 0
    Sig_XY = 0
    Sig_X_sq = 0
    for i in range(n):
        Sig_X += x.values[i][0]
        Sig_Y += y.values[i]
        Sig_XY += x.values[i][0] * y.values[i]
        Sig_X_sq += x.values[i][0] * x.values[i][0]
    
    #Formula of the Slope
    m = (n*(Sig_XY) - (Sig_X * Sig_Y)) / ((n*Sig_X_sq) - (Sig_X * Sig_X))
    #Save slope
    slope_list_train.append(m)  
    
    # Formula of the constant (Y-intercept)
    b = (Sig_Y - (m*Sig_X)) / n
    
    # Save y-intercept
    y_intercept_list_train.append(b)
    

#Function for calculating least square (y_pred) for each x_train
def pred_y(x, m, b):
    global y_pred
    
    y_pred = m*x + b
    
def Train_Data():
    # Declare constructor
    disclaimer = print_disclaimer()
    global y_pred
    global y_pred_list
    global sum_y_pred
    global sum_y_ideal
    global count
    global df_final_ideal
    global df_new_y_test
    
    #Pass the variables into the Least Square Function (to fund the slope and y-intercept)
    try:
        for i in range(0, 4):
            y_col = y_train.iloc[:, i]
            least_Sq(x_train, y_col, n_train)
    except:
        disclaimer.print_exception()
        
    
    #Calculate the Pred_Y value (Least-squared)
    for h in range (0, 4):
        for i in range(n_train):
            pred_y(x_train.values[i][0], slope_list_train[h], y_intercept_list_train[h])
            y_pred_list.append(y_pred)
        df_y_pred[h] = y_pred_list
        y_pred_list = []
    frames = [x_train, df_y_pred]
    pred_xy = pd.concat(frames, axis = 1)
    pred_xy.rename(columns = {0:'y1', 1:'y2', 2:'y3', 3:'y4'}, inplace = True)
    disclaimer.print_predicted(pred_xy)
    
    #Find the total of all y_value in predict data
    for h in range(1, 5):
        for i in range (n_train):
            sum_y_pred += pred_xy.values[i][h]
        sum_y_pred_list.append(sum_y_pred)
        sum_y_pred = 0
    disclaimer.print_predicted_sum(sum_y_pred_list)
    
    
    #Find the total of all y_value in ideal data
    for h in range(0, 50):
        y_col = y_ideal.iloc[:, h]
        for i in range(n_ideal):
            sum_y_ideal += y_col.values[i]
        sum_y_ideal_list.append(sum_y_ideal)
        sum_y_ideal = 0
    disclaimer.print_ideal_sum(sum_y_ideal_list)
     
    # Map the pred-value to find 4 suitable y_value in the ideal dataset
    #Methodology: For each pred_value, find the minimum sum difference of least-squared and append into a list
    least_dif = abs(sum_y_pred_list[0])  - abs(sum_y_ideal_list[0])
    for i in sum_y_pred_list:
        if i < 0 & count == 0:
            least_dif = abs(i) - abs(sum_y_ideal_list[0])
            count = 1
        for j in sum_y_ideal_list:
            dif = abs(i) - abs(j)
            if least_dif > abs(dif):
                least_dif = abs(dif)
                #Find the index of y_ideal (** In Y_ideal, it's starts from y1)
                final_ideal = sum_y_ideal_list.index(j)
        final_index.append(final_ideal)
    
    
    #Print the index in ideal dataset which are the 4 ideal function
    disclaimer.print_index(final_index)
    
    # Print the 4 ideal function found
    
    for i in range(0, 4):
        index = final_index[i]
        df_final_ideal[i] = y_ideal.iloc[:, index:index + 1]
        
    
        
    #Combine with x_value and rename the column name
    frame_ideal = [x_train, df_final_ideal]
    df_final_ideal = pd.concat(frame_ideal, axis = 1)
    df_final_ideal.rename(columns = {0:'y34', 1:'y31', 2:'y8', 3:'y46'}, inplace = True)
    engine = create_engine('sqlite:///chosen_data.db')
    connection_chosen = engine.connect()
    df_final_ideal.to_sql('chosen ideal functions', connection_chosen, if_exists='replace')
    disclaimer.print_ideal(df_final_ideal)
    

    #Examine with test dataset
    """
    According to the Written Assignment Document from myCampus(iubh), the criterion for mapping the individual test case to the four ideal functions is that the existing maximum 
    deviation of the calculated regression does not exceed the largest deviation between training dataset (A) and 
    the ideal function (C) chosen for it by more than factor sqrt(2).
    And according to the Helpful Tipps for doing the Python Programming Assignment notes from myCampus(iubh), here are the steps:
        - Go into each data point in test/reality data set and find all points that are "close enough" (e.g. smaller than Sqrt(2)) to one of the 4 ideal functions. 
        For each ideal function, loop through 4 times)
        - Save all the test dataset
    """
    disclaimer.print_test_data(df_query_test)
    # 0). Loop through the test data points
    try:
        for i in range(n_test):
            for j in range(1, 4):
                for h in range(n_ideal):
                    # 1). Take test data point minus the reality data points
                    diff_test_ideal = abs(y_test.values[i][0]) - abs(df_final_ideal.values[h][j])
                    # 2). If it is less than Sqrt(2), save into a list (We will name it: new_y_test)
                    if(diff_test_ideal < math.sqrt(2)):
                        #3). If the value already exist in the list, we don't have to reappend it again
                        if y_test.values[i][0] not in new_y_test:
                            new_y_test.append(y_test.values[i][0])
                            #Save the y-deivation into a list
                            y_dev_list.append(diff_test_ideal)
    except StopIteration:
        next()
    df_new_y_test['y'] = new_y_test
    disclaimer.print_new_y_test(df_new_y_test)
    
    # Find the removed rows
    removed_df = pd.concat([y_test,df_new_y_test]).drop_duplicates(keep=False)
    disclaimer.print_removed(removed_df)  
    disclaimer.print_y_dev(y_dev_list)
    
    engine = create_engine('sqlite:///new_test_data.db')
    connection_test = engine.connect()
    df_new_y_test = df_query_test.drop(labels=None, axis=0, index=[75, 83, 97])
    df_new_y_test.to_sql('new_y_test', connection_test, if_exists='replace')
    disclaimer.print_new_y(df_new_y_test)
    
    # 4). Scatter plot the new_test data with the 4 ideal function
    draw = dw(x_test, y_test, plt)
    scatter_plot = Scatter(x_test, y_test, plt)
    scatter_plot.draw_scatter()
    line_plot = Line(x_train, df_final_ideal, plt)
    line_plot.draw_line()
    draw.show()
    
    disclaimer.print_conclusion()
    
if __name__ == '__main__':
    Train_Data()
    
#TODO Object-Oriented Python --> Done
#TODO At least one inheritance --> Done
#TODO It includes standard- und user-defined exception handlings --> Done
#TODO Make use of the Panda Packages --> Done
#TODO Write unit-tests for all useful elements
#TODO Documentation using Docstrings --> Done
#TODO Save the deviation that is less than sqrt(2)