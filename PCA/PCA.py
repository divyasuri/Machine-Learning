'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import pandas as pd #import necessary packages 
import numpy as np
import math

def csv_writer(filename, output_file_name): #this function preprocesses the txt file provided and outputs a csv file with all the data split into columns 
    output_file = open(output_file_name,"w") #open output_file_name in write only mode
    with open(filename,'r') as inp:
        data = inp.readlines() #split input file into lines
        for line in range(len(data)):
            insert_data = data[line].split('\t') #split each line by tab space
            for value in insert_data: 
                value = value.rstrip()
                output_file.write(str(value))
                output_file.write(',')
            output_file.write('\n') #write data into csv file
        output_file.close()

def pca_implementation(): #this function runs the algorithm 
    csv_writer('pca-data.txt','PCAData.csv') #preprocessing step to generate the csv from which data will be analyzed
    pca_data = pd.read_csv('PCAData.csv',header=None) #read csv into a pandas dataframe
    pca_data.rename({3:"a"}, axis="columns",inplace=True) #drop redundant column 
    pca_data.drop(["a"],axis=1,inplace=True)
    pca_data.columns=['x','y','z'] #name each column with these labels
    pca_matrix = pd.DataFrame.as_matrix(pca_data) #convert dataframe into a matrix
    column_means = np.mean(pca_matrix.T,axis = 1) #calculate the means for each column, .T forces the calculation by column
    centered = pca_matrix - column_means #center values in each column by subtracting column means 
    cov_mat = np.cov(centered.T) #generate the covariance matrix of the centered matrix
    eig_values,eig_vectors = np.linalg.eig(cov_mat) #compute eigenvalues and corresponding eigenvectors 
    eig_pairs = [] #initialize an empty list of paired eig value and eig vectors 
    for i in range(len(eig_values)):
        eig_tuple = (eig_values[i],eig_vectors[:,i]) #create a tuple of the eig value and its corresponding eig vectors
        eig_pairs.append(eig_tuple) #append to list 
    eig_pairs.sort(key = lambda x:x[0],reverse=True) #sort eig pairs by decreasing eig values, to construct the final 2D matrix we will remove the smallest eig value 
    eig_vector_matrix = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1))) #extract the eig vectors for the top 2 eig values and create a reduced matrix by reshaping each to 3 rows and 1 column 
    transformation = np.dot(centered,eig_vector_matrix) #transform centered dataset to new dimensional space using 
    print('Direction of Principal Component 0: ' + str(eig_vector_matrix[:,0])) #print directions
    print('Direction of Principal Component 1: ' + str(eig_vector_matrix[:,1])) 
    output_txt = open("TransformedData.txt","w") #create an output txt file 
    for point,point_transformed in zip(pca_matrix,transformation): 
        output_txt.write('From {0} to {1}'.format(point,point_transformed)) #writes each original point and its transformation 
        output_txt.write('\n')
    output_txt.close()

pca_implementation()


