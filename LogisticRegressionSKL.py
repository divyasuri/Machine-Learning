'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import pandas as pd #import necessary modules
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def csv_writer(filename, output_file_name): #this function preprocesses the txt file provided and outputs a csv file with all the data split into columns 
    output_file = open(output_file_name,"w") #open output_file_name in write only mode
    with open(filename, 'r') as inp: 
        data = inp.readlines() #split input file into lines
        for line in range(len(data)):
            insert_data = data[line].split('\t') #split each line by tab space
            for value in insert_data:
                value =  value.rstrip()
                output_file.write(str(value))
                output_file.write(',')
            output_file.write('\n') #write data into csv file
        output_file.close()

def implementation(): #this function runs the algorithm 
    csv_writer('classification.txt','LRData.csv') #preprocessing step to generate the csv from which data will be analyzed
    LR_data = pd.read_csv('LRData.csv',header=None) #read csv into a pandas dataframe
    LR_data.drop([5],axis="columns",inplace=True) #drop redundant column 
    LR_data.columns=['x','y','z','drop','labels'] #name each column with these labels
    LR_data.drop(['drop'],axis="columns",inplace=True) #drop 4th column 
    x_values = LR_data.iloc[:,0:3].values #subset data into the columns without labels
    y_values = LR_data.iloc[:,3].values #subset data into column with labels
    log_reg = LogisticRegression() #initiate instance of the algorithm my sklearn
    log_reg.fit(x_values, y_values) #fit the instance to our data
    pred = log_reg.predict(x_values) #use model instance to predict probabilities of each classification
    accuracy = accuracy_score(y_values.flatten(),pred) #retrieve accuracy score
    weights = log_reg.coef_ #retrieve weights
    intercept = log_reg.intercept_ #retrieve intercept
    print('Weights: ')
    print(weights)
    print('Intercept: ' + str(intercept))
    print('Accuracy: ' + str(accuracy))

implementation()