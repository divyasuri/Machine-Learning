import pandas as pd #import required packages and libraries 
from sklearn import tree
import os.path 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
import numpy as np

def csv_writer(filename, output_file_name): #this function takes in a text file and converts it into a CSV 
    headers = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy'] #set list of headers for this CSV
    output_file = open(output_file_name,"w") #opening the output file in write mode
    fileEmpty = os.stat(output_file_name).st_size == 0 #checking if the file is empty
    if fileEmpty: #if the file is empty, write in the headers
        for header in headers:
            output_file.write(header)
            output_file.write(',')
        output_file.write('\n')
    with open(filename, 'r') as inp:
        data = inp.readlines()
        for line in range(len(data)):
            if data[line] == 0 or data[line] == 1: #skipping the first and second line of the txt file which are empty 
                continue
            else:
                insert_data = data[line].split(',') #split each line in the txt file by commas
                insert_data[-1] = insert_data[-1].strip()
                insert_data[-1] = insert_data[-1].strip(';')
                insert_data[0] = insert_data[0].split(':') #remove the columns with ':' 
                del insert_data[0][0] #remove unnecessary columns
                insert_data[0] = ''.join(insert_data[0]) 
                for value in insert_data:
                    value =  value.lstrip() #remove empty whitespace
                    output_file.write(str(value)) #write values to the open CSV file
                    output_file.write(',')
                output_file.write('\n') 
        output_file.close()

csv_writer('dt-data.txt','Decision_Tree_Data.csv') #creates a CSV called 'Decision_Tree_Data.csv' in the directory of the script

enjoy_data = pd.read_csv('Decision_Tree_Data.csv',index_col=None,header=0) #Using pandas to the read the CSV into a dataframe 
enjoy_data.rename({"Unnamed: 7":"a"}, axis="columns",inplace=True) #remove last column 
enjoy_data.drop(["a"],axis=1,inplace=True)
enjoy_data.drop([0,1],axis=0, inplace=True)
enjoy_data.reset_index(drop=True, inplace=True)

def occupied_label(row): #function to convert occupied column values into integers corresponding to attribute values
    if row["Occupied"] == 'Low':
        return 0
    if row["Occupied"] == 'Moderate':
        return 1
    if row["Occupied"] == 'High':
        return 2

def price_label(row): #function to convert price column values into integers corresponding to attribute values
    if row["Price"] == 'Cheap':
        return 0
    if row["Price"] == 'Normal':
        return 1
    if row["Price"] == 'Expensive':
        return 2

def music_label(row): #function to convert music column values into integers corresponding to attribute values
    if row["Music"] == 'Quiet':
        return 0
    if row["Music"] == 'Loud':
        return 1

def location_label(row): #function to convert location column values into integers corresponding to attribute values
    if row["Location"] == 'Talpiot':
        return 0
    if row["Location"] == 'City-Center':
        return 1
    if row["Location"] == 'German-Colony':
        return 2
    if row["Location"] == 'Ein-Karem':
        return 3
    if row["Location"] == 'Mahane-Yehuda':
        return 4

def VIP_label(row): #function to convert VIP column values into integers corresponding to attribute values
    if row["VIP"] == 'No':
        return 0
    if row["VIP"] == 'Yes':
        return 1
    
def beer_label(row): #function to convert Favorite Beer column values into integers corresponding to attribute values
    if row["Favorite Beer"] == 'No':
        return 0
    if row["Favorite Beer"] == 'Yes':
        return 1

def enjoy_label(row): #function to convert Enjoy column values into integers corresponding to attribute values
    if row["Enjoy"] == 'No':
        return 0
    if row["Enjoy"] == 'Yes':
        return 1

enjoy_data["Occupied"] = enjoy_data.iloc[:,0:].apply(lambda row: occupied_label(row),axis=1) #apply all these functions to their respective columns
enjoy_data["Price"] = enjoy_data.iloc[:,1:].apply(lambda row: price_label(row),axis=1)
enjoy_data["Music"] = enjoy_data.iloc[:,2:].apply(lambda row: music_label(row),axis=1)
enjoy_data["Location"] = enjoy_data.iloc[:,3:].apply(lambda row: location_label(row),axis=1)
enjoy_data["VIP"] = enjoy_data.iloc[:,4:].apply(lambda row: VIP_label(row),axis=1)
enjoy_data["Favorite Beer"] = enjoy_data.iloc[:,5:].apply(lambda row: beer_label(row),axis=1)
enjoy_data["Enjoy"] = enjoy_data.iloc[:,6:].apply(lambda row: enjoy_label(row),axis=1)

X = enjoy_data.values[:,0:6] #split the dataset into x-values and y-values
Y = enjoy_data.values[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) #using scikit learn's train_test_split method, split above data set into training and test set

enjoy_data_classifier = DecisionTreeClassifier(criterion = "entropy") #set up classifier

enjoy_data_classifier.fit(X_train,Y_train) #fit classifier to training data

Y_pred = enjoy_data_classifier.predict(X_test) #use classifier to predict test set y-value

test_accuracy = accuracy_score(Y_test,Y_pred) #assess accuracy of classifier

data_to_predict_on = np.array([1,0,1,1,0,0]) #create array to predict outcome of (Occupied=Moderate; price=Cheap; music=Loud; location=City-Center; VIP=No; favorite beer=No)
data_to_predict_on = data_to_predict_on.reshape(1,-1)

predicted = enjoy_data_classifier.predict(data_to_predict_on) #predicted value

if predicted == 0: #convert prediction to categorical label 
	print('No')
if predicted == 1:
	print('Yes')
