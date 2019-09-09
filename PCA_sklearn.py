'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import pandas as pd #import necessary packages 
from sklearn.decomposition import PCA

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

def pca_sklearn():
    pca = PCA(2) #create an instance for the PCA algorithm with k = 2
    pca_data = pd.read_csv('PCAData.csv',header=None) #read csv into a pandas dataframe
    pca_data.rename({3:"a"}, axis="columns",inplace=True) #drop redundant column 
    pca_data.drop(["a"],axis=1,inplace=True)
    pca_data.columns=['x','y','z'] #name each column with these labels
    pca_matrix = pd.DataFrame.as_matrix(pca_data) #convert dataframe into a matrix
    pca.fit(pca_matrix) #fit created instance to original data
    sklearn_transformation = pca.transform(pca_matrix) #create transformations of the original data
    vectors=pca.components_ #extract the directions
    print('SK Learn Direction of Principal Component 0: ' + str(vectors[0])) #print directions
    print('SK Learn Direction of Principal Component 1: ' + str(vectors[1]))  
    output_txt_file = open("SKLTransformedData.txt","w") #create an output txt file 
    for p,pt in zip(pca_matrix,sklearn_transformation): 
        output_txt_file.write('From {0} to {1}'.format(p,pt)) #writes each original point and its transformation 
        output_txt_file.write('\n')
    output_txt_file.close()

pca_sklearn()