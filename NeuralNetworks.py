'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import numpy as np #import necessary modules 

def load(pgm):
    with open(pgm, 'rb') as f:
        f.readline()   # skip the magic number
        f.readline()   #skip the whitespace
        width, height = f.readline().split()  #get the dimension of the image
        width = int(width)
        height = int(height) #convert them into numbers
        scale = int(f.readline().strip())       
        image = []
        for i in range(width*height):
            image.append((f.read(1)[0]) / scale)   
        return image


class NeuralNetwork(object): #implements an instance of the Neural Network algorithm 

	def __init__(self,lr=0.1,epochs=1000,input_neurons = 960,hidden_neurons = 100, output_neurons = 1): #initialize each instance with these parameters 
		self.epochs = epochs
		self.lr = lr
		self.input_neurons = input_neurons
		self.hidden_neurons = hidden_neurons
		self.output_neurons = output_neurons

	def train(self,train_images_array,train_labels_array): #this function trains the algorithm with the data provided to find the optimal weights
		self.hidden_weights=np.random.uniform(low=-0.01,high=0.01,size=(self.input_neurons,self.hidden_neurons)) #set weights and bias for the hidden layer 
		self.hidden_bias=np.random.uniform(size=(1,self.hidden_neurons)) 
		self.output_weights=np.random.uniform(low=-0.01,high = 0.01,size=(self.hidden_neurons,self.output_neurons)) #set weights and bias for the output layer
		self.output_bias=np.random.uniform(size=(1,self.output_neurons))
		for i in range(self.epochs): #run through 1000 iterations 
		    hidden_inp=np.dot(train_images_array,self.hidden_weights) #start forward propogation 
		    hidden_inp=hidden_inp + self.hidden_bias
		    hidden_activation = self.SigmoidFunction(hidden_inp) #activation for the hidden layer
		    output_inp=np.dot(hidden_activation,self.output_weights)
		    output_inp= output_inp + self.output_bias
		    output = self.SigmoidFunction(output_inp) #activation for the output layer
		    output_error = train_labels_array-output #start the back propogation
		    output_slope = self.SigmoidDer(output)
		    hidden_slope = self.SigmoidDer(hidden_activation) 
		    output_delta = output_error * output_slope
		    hidden_error = np.dot(output_delta,self.output_weights.T)
		    hidden_delta = hidden_error * hidden_slope
		    self.output_weights += np.dot(hidden_activation.T,output_delta) *self.lr #adjust output layer weights and bias
		    self.output_bias += np.sum(output_delta, axis=0) *self.lr
		    self.hidden_weights += np.dot(train_images_array.T,hidden_delta) *self.lr  #adjust hidden layer weights and bias
		    self.hidden_bias += np.sum(hidden_delta, axis=0) *self.lr
		return self

	def SigmoidFunction(self,x): #this function calculates the sigmoid function
		return 1.0 / (1.0 + np.e ** (-x))

	def SigmoidDer(self, x): #this function calculates the derivative to obtain the slope at each layer 
		return self.SigmoidFunction(x) *(1-self.SigmoidFunction(x))

	def prediction(self,test_images_array): #this function generates the final prediction based on optimal weights
		inp = np.dot(test_images_array,self.hidden_weights) #activation in hidden layer
		inp = inp + self.hidden_bias
		inp_activation = self.SigmoidFunction(inp)
		out_inp = np.dot(inp_activation,self.output_weights)
		out_inp = out_inp + self.output_bias
		out = self.SigmoidFunction(out_inp) #activation in output layer
		return np.where(out >= 0.5, 1, 0) #converts output back to 1/0 range

def implementation(): #this function runs the algorithm 
	train_images = [] #initialize list for attributes of each training image
	train_labels = [] #initialize list for assigned labels for the training image based on name of image
	with open('downgesture_train.list') as f:
	    for train_image in f.readlines():
	        train_image = train_image.strip() #remove new line character from the string 
	        train_images.append(load(train_image)) #append list of attributes for individual image to larger list
	        if 'down' in train_image:
	            train_labels.append(1) #assign 1 as label if 'down' in name
	        else:
	            train_labels.append(0)
	test_labels = [] #initialize list for assigned labels for the test image based on name of image
	test_images = [] #initialize list for attributes of each test image
	with open('downgesture_test.list') as g:
	    for test_image in g.readlines(): #iterate through test images
	        test_image = test_image.strip()
	        test_images.append(load(test_image)) #append list of attributes for individual image to larger list
	        if 'down' in test_image:
	            test_labels.append(1) #assign 1 as label if 'down' in name
	        else:
	            test_labels.append(0)
	train_images_array = np.asarray(train_images) #convert list to array
	train_labels_array = np.asarray(train_labels)
	train_labels_array = train_labels_array.reshape(len(train_labels),1) #reshape array to 1D
	test_images_array = np.asarray(test_images)
	test_labels_array = np.asarray(test_labels)
	test_labels_array = test_labels_array.reshape(len(test_labels),1)
	model = NeuralNetwork().train(train_images_array,train_labels_array) 
	pred = model.prediction(test_images_array)
	print(pred)
	correct = np.where(test_labels_array==pred)[0].shape[0] #compare predicted output with actual outputs
	total = pred.shape[0]
	accuracy = (correct / total) * 100 #accuracy score
	print('Correct Classification: ' + str(correct))
	print('Accuracy Score: ' + str(accuracy) + '%')

implementation()




