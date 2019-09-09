'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import pandas as pd #import necessary modules
import numpy as np
import math

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

class LogisticRegression(object): #implements an instance of logistic regression

	def __init__(self,learning_rate=0.1,iterations=7000): #initialize each instance with these parameters (7000 as max iterations, 0.1 as learning rate)
		self.learning_rate = learning_rate
		self.iterations = iterations

	def train(self,x_values,y_array): #this function trains the algorithm with the data provided to find the optimal weights
		x_values = np.insert(x_values, 0, 1, axis = 1) #add the intercept which increases dimensions from 3 to 4
		length,dimensions = np.shape(x_values) #retrieve shape of the x_values
		self.weights = np.random.rand(dimensions) #initialize an array for the weights starting with random values
		for i in range(self.iterations): #run through 7000 iterations
			inp = np.dot(x_values,self.weights) #compute dot product of weights and x values
			output = self.SigmoidFunction(inp) #call on sigmoid function to generate probability
			errors = y_array - output #calculate errors from actual values
			self.weights += self.learning_rate/length * (np.dot(errors,x_values)) #adjust weights accordingly 
		return self

	def SigmoidFunction(self,x): #this function calculates the sigmoid function
		return 1.0 / (1.0 + np.e ** (-x))

	def prediction(self,x_values): #this function generates the final prediction based on optimal weights
		x_values = np.insert(x_values, 0, 1, axis = 1)
		input_value = np.dot(x_values,self.weights)
		out = self.SigmoidFunction(input_value)
		return np.where(out >= 0.5, 1, -1) #converts output back to 1/-1 range


def implementation(): #this function runs the algorithm 
	csv_writer('classification.txt','LRData.csv') #preprocessing step to generate the csv from which data will be analyzed
	LR_data = pd.read_csv('LRData.csv',header=None) #read csv into a pandas dataframe
	LR_data.drop([5],axis="columns",inplace=True) #drop redundant column 
	LR_data.columns=['x','y','z','drop','labels'] #name each column with these labels
	LR_data.drop(['drop'],axis="columns",inplace=True) #drop 4th column 
	x_values = LR_data.iloc[:,0:3].values #subset data into the columns without labels
	y_test = LR_data.iloc[:,3].values
	y_values = LR_data.iloc[:,3].values #subset data into column with labels
	y_values= ([0 if x == -1 else x for x in y_values]) #convert labels from 1/-1 to 1/0 range
	y_array = np.asarray(y_values) #convert the list into an array
	model = LogisticRegression().train(x_values,y_array) #initialize an instance of the LogisticRegression object and train it 
	pred = model.prediction(x_values) #invoke the prediction method on the instance
	print('Weights: ')
	print(model.weights)
	correct = np.where(y_test==pred)[0].shape[0] #compare predicted output with actual outputs
	total = pred.shape[0]
	accuracy = correct / total #accuracy score
	print('Correct Classification: ' + str(correct))
	print('Total Data Points: ' + str(total))
	print('Accuracy Score: ' + str(accuracy))

implementation()
