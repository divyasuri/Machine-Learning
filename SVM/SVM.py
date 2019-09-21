'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import pandas as pd #import necessary modules
import numpy as np
import math
import cvxopt

class SVM(object): #implements an instance of the SVM algorithm

	def __init__(self,threshold = 0.00001): #initialize each instance with these parameters
		self.weights = None
		self.bias = None
		self.support_vectors = None
		self.threshold = threshold


	#train the SVM
	def train(self,x_values,y_values): #this function trains the algorithm with the data to find the parameters of the equation 
		length,dimensions = np.shape(x_values) 
		K = y_values.reshape(length,1)
		K = K * x_values
		K = np.dot(K,K.T)
		K = cvxopt.matrix(K) #kernel 
		ones = cvxopt.matrix(np.ones(length) * -1) #vector of ones with length of y values
		zeros = cvxopt.matrix(np.zeros(length)) #vector of zeros with length of y values
		yT = cvxopt.matrix(y_values,(1,length)) #transpose of the y values
		b = cvxopt.matrix(np.zeros(1)) #matrix with just one value, a zero
		identity = cvxopt.matrix(np.eye(length)*-1) #identity matrix with -1 as diagonals 
		solver = cvxopt.solvers.qp(K, ones, identity, zeros, yT, b) #quadratic programming solver
		alphas = np.array(solver['x']) #obtain alphas from solver
		self.support_vectors = (np.argwhere(alphas > self.threshold)).transpose()[0] #support vectors have alphas greater than threshold 
		self.weights = np.sum(alphas * (y_values.reshape(length,1)) * x_values, axis = 0) #compute weights 
		condition = (alphas > self.threshold).reshape(-1)
		b = y_values[condition] - np.dot(x_values[condition], self.weights) #compute bias
		self.bias = b[0] 

	def predict(self,x_values): #this function uses the computed weights and biases to predict y values from input x values
		predicted = [] 
		for x in x_values:
			pred = np.dot(x,self.weights) + self.bias #equation of the line 
			predicted.append(pred)
		results = []
		for p in predicted: # sign of y = x.w + b for classification 
			if p < 0:
				results.append(-1)
			else:
				results.append(1)
		return results

def implementation(): #this function runs the algorithm 
	data = pd.read_csv("linsep.txt",delimiter=',',header=None)
	data.columns = ["x", "y", "label"]
	x_values = data.iloc[:,0:2].values #subset data into the columns without labels
	y_values = data.iloc[:,2].values
	y_values = [float(v) for v in y_values] #convert y values to float
	y_values = np.asarray(y_values) #convert the list into an array
	model = SVM() #start an instance of the model 
	model.train(x_values,y_values) #train the model 
	print('Weights: ')
	print(model.weights)
	print('Bias: ') 
	print(model.bias)
	pred = model.predict(x_values) #generate predictions

implementation()
