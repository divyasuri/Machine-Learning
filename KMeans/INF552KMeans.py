'''
    Group Members: 
                        Divyata Singh 
                        Divya Suri
                        Feng Qiu
'''

import pandas as pd #import all necessary modules
import os.path 
import numpy as np
import math

def csv_writer(filename, output_file_name): #this function preprocesses the txt file provided and outputs a csv file with all the data split into columns of x and y values
    headers = ['X','Y'] #headers for the CSV file 
    output_file = open(output_file_name,"w") 
    fileEmpty = os.stat(output_file_name).st_size == 0 #check if the file is empty 
    if fileEmpty: #if file is empty, write the headers into the file
        for header in headers:
            output_file.write(header)
            output_file.write(',')
        output_file.write('\n')
    with open(filename, 'r') as inp: #open input file in read only mode
        data = inp.readlines() #read through each line of the input file and store as data
        for line in range(len(data)):
            insert_data = data[line].split(',') #split each line in data by commas
            for value in insert_data:
                value =  value.rstrip() #remove white space from each value
                output_file.write(str(value)) #write each value into rows and columns of the output file
                output_file.write(',')
            output_file.write('\n')
        output_file.close()

def EuclideanDistanceCalc(sample1, sample2): #this function calculates the euclidean distance between two points
    for value in sample1: #the input values are two sets of tuples, therefore the x and y values need to be extracted 
        x1 = sample1[0]
        y1 = sample1[1]
    for value in sample2:
        x2 = sample2[0]
        y2 = sample2[1]
    sub1 = x2 - x1 
    sub2 = y2 - y1
    sqrd_dist1 = sub1**2
    sqrd_dist2 = sub2**2
    summed = sqrd_dist1 + sqrd_dist2
    euc_dis = math.sqrt(summed)
    return euc_dis #returns euc distance between both points

def RandomCentroids(data, num_of_centroids): #this function generates random samples from the list of clusters as the initial centroids 
    centroids = [] #initialize empty list for each centroid value
    centroids_index = np.random.choice(len(data), size=num_of_centroids, replace=False) #choose random values as centroids, this returns the index of those values in the list of possible values
    for i in centroids_index:
        centroid = data[i] #associated value with each index chosen at random
        centroids.append(centroid)
    return centroids #returns list of random values as centroids

def DistanceCalc(samples, centroid_dict): #this function calculates the distances between the data points and the centroids
    distance_arrays = list() #initialize an empty list for the distances
    for sample in samples: #loop through all the sample values given
        for key,value in centroid_dict.items(): #loop through the values associated with each centroid 
            distance = EuclideanDistanceCalc(sample,value) #calculate euc distance between the sample values and the centroids
            distance_arrays.append([key, tuple(sample), distance]) #store the centroid, sample and distance as one value and append to the main list
    return distance_arrays

def DistanceTable(distance_arrays): #this function generates a distance table between each sample value and the centroid
    labels = ['Centroid','Sample','Distance from Centroid'] #labels for the table
    distance_table = pd.DataFrame.from_records(distance_arrays, columns=labels) #build a pandas dataframe from the distance array
    distance_table = distance_table.pivot(index='Sample', columns='Centroid', values='Distance from Centroid') #rearrange the table to make each sample value the index
    return distance_table

def ClusterBuilder(distance_table): #this function assigns each sample value to the centroid it is closest to by euc distance
    M1_cluster = list() #initialize a list for the 3 clusters
    M2_cluster = list()
    M3_cluster = list()
    for index, row in distance_table.iterrows(): #loop through each row of the distance table
        cluster_dict = dict() #initialize a dictionary for each row
        cluster_dict['Sample'] = index #assign sample to the sample value
        min_value = row.idxmin(axis=1) #identify the cluster by finding the minimum distance and its associated value
        cluster_dict['Cluster'] = min_value 
        if min_value == 'M1': #append each individual dictionary to their corresponding cluster's list
            M1_cluster.append(cluster_dict)
        elif min_value == 'M2':
            M2_cluster.append(cluster_dict)
        elif min_value == 'M3':
            M3_cluster.append(cluster_dict)
    return M1_cluster,M2_cluster,M3_cluster

def NewCentroids(M1_cluster,M2_cluster,M3_cluster): #this function calculates the new centroids after each sample value is assigned to a cluster
    new_centroid_dict = dict() #initialize a dictionary with all the new centroids
    x_sum_M1 = 0 #initialize x and y values for each centroid
    y_sum_M1 = 0 
    x_sum_M2 = 0 
    y_sum_M2 = 0
    x_sum_M3 = 0 
    y_sum_M3 = 0
    for item in M1_cluster: #loop through each sample value in each cluster and calculate sum of x and y values
        x_sum_M1 = x_sum_M1 + item['Sample'][0]
        y_sum_M1 = y_sum_M1 + item['Sample'][1]
    for item in M2_cluster:
        x_sum_M2 = x_sum_M2 + item['Sample'][0]
        y_sum_M2 = y_sum_M2 + item['Sample'][1]
    for item in M3_cluster:
        x_sum_M3 = x_sum_M3 + item['Sample'][0]
        y_sum_M3 = y_sum_M3 + item['Sample'][1]
    new_centroid_dict['M1'] = [(x_sum_M1/len(M1_cluster)),(y_sum_M1/len(M1_cluster))] #assign new centroid values, the averages of x and y values
    new_centroid_dict['M2'] = [(x_sum_M2/len(M2_cluster)),(y_sum_M2/len(M2_cluster))]
    new_centroid_dict['M3'] = [(x_sum_M3/len(M3_cluster)),(y_sum_M3/len(M3_cluster))]
    return new_centroid_dict

def k_means(): #this function runs the algorithm 
	max_iterations = 1000 #setting max iterations 
	iterations = 0 #initializing iterationc count
	csv_writer('clusters.txt','Clusters.csv') #preprocessing step to generate the csv from which data will be analyzed
	clusters_data = pd.read_csv('Clusters.csv',index_col = None, header=0) #read csv into a pandas dataframe
	clusters_data.rename({"Unnamed: 2":"a"}, axis="columns",inplace=True) #drop redundant column 
	clusters_data.drop(["a"],axis=1,inplace=True)
	clusters_list = clusters_data.values.tolist() #convert pandas dataframe into a list of values
	initial_centroids = RandomCentroids(clusters_list,3) #generate 3 random centroids from the list of sample values
	initial_centroid_dict = {} #create a dictionary assigning each random value to their own cluster
	initial_centroid_dict['M1'] = initial_centroids[0]
	initial_centroid_dict['M2'] = initial_centroids[1]
	initial_centroid_dict['M3'] = initial_centroids[2]
	initial_distances = DistanceCalc(clusters_list,initial_centroid_dict) #calculate euc distance between sample values and these initial centroids
	initial_distances_table = DistanceTable(initial_distances) #create distance table from initial distances
	M1_cluster,M2_cluster,M3_cluster = ClusterBuilder(initial_distances_table) #assign each sample to a cluster
	iterations = iterations + 1 #adding to iterations 
	new_centroids = NewCentroids(M1_cluster,M2_cluster,M3_cluster) #generate new centroids from the sample values in each cluster
	new_distances = DistanceCalc(clusters_list,new_centroids) #re-do process of assignment with these new centroids
	new_distances_table = DistanceTable(new_distances)
	new_cluster1, new_cluster2, new_cluster3 = ClusterBuilder(new_distances_table)
	iterations = iterations + 1 #adding to iterations
	next_centroids = NewCentroids(new_cluster1, new_cluster2, new_cluster3) #after assignment, recalculate centroids
	while next_centroids != new_centroids and iterations <= max_iterations: #set up while loop that keeps running if the two centroid values after two iterations are not the same and max iteration count has not been reached, therefore assignments were changed
		new_centroids = next_centroids #establish new centroid as the calculate centroids from previous iteration 
		new_distances = DistanceCalc(clusters_list,new_centroids) #re-do steps for assignment and recalculation of centroids
		new_distances_table = DistanceTable(new_distances)
		new_cluster1, new_cluster2, new_cluster3 = ClusterBuilder(new_distances_table)
		iterations = iterations + 1 #add to iterations with each loop
		next_centroids = NewCentroids(new_cluster1, new_cluster2, new_cluster3)	
	print('The initial centroids used were : ' + '\n' + str(initial_centroid_dict['M1']) + '\n' + str(initial_centroid_dict['M2']) + '\n' + str(initial_centroid_dict['M3']) + '\n')
	print('Final Clusters Centroids:' + '\n' + str(next_centroids['M1']) + '\n' + str(next_centroids['M2']) + '\n' + str(next_centroids['M3'])) #once break point is reached, the output is the final centroids for each cluster
	print('\n' + 'The number of iterations required till convergence were: ' + str(iterations))

k_means()
