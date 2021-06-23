# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:27:30 2021

@author: Gebruiker
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import Pathfinding_algorithms_functions as pfa
import time

#800 point layout:
file_name = 'array_400_linker_test.cif'

def read_file(file):
    '''Create list of features from txt or cif file. Structure of input file: 
    square feature: ['B', x_length, y_length, xpos, ypos] 
    circle feature: ['R', radius, xpos, ypos]. Returns a list of lists: features_final.'''
    features_file = open(file)
    features_lines = features_file.readlines()
    features=[]
    for line in features_lines:
        #only count lines which contain a feature and strip line
        if (line.startswith('B') or line.startswith('R')):
            line=line.rstrip()
            line=line.strip(';')
            line = line.split()
            features.append(line)
    features_final=[]
    for feature in features:
        #turn all integers into floats
        numbers = feature[1:]
        numbers_floats=[]
        for number in numbers:
            numbers_floats.append(float(number))
        features_final.append([feature[0]]+numbers_floats)
    return features_final
  
features_final = read_file(file_name)

features_final_mirror = []

#Make use of symmetry and only use the left half of the layout
for feature in features_final:
    if feature[0]=='B':
        if feature[3]<0:
            features_final_mirror.append(feature)
    if feature[0]=='R':
        if feature[2]<0:
            features_final_mirror.append(feature)

#define all dimensions in nm
width = 100000000 #in nm,10cm
size_of_feature = 45000 #in nm, 45 um
resolution = width//size_of_feature
radius_wafer = 50000000 #in nm, 5cm
dimensions = [width, resolution, radius_wafer]            

def discretize(features, dimensions):
    '''Create a total_feature_array, containing all circles and squares in features. Uses input features
    and dimensions to determine size of array. Structure of dimensions is [length, resolution, radius_wafer].
    Also calculates the centre of each element and returns a dictionary with these centres as keys and the rest of the 
    information on the feature as the value. Returns the total_feature_array and dictionary centres'''
    #initialise time
    t=time.time()
    #extract features from dimensions list
    length = dimensions[0]
    resolution = dimensions[1]
    radius_wafer = dimensions[2]    
    #create empty arrays based on dimensions
    x_coords = np.linspace(-(length/2), (length/2), num=resolution)
    y_coords = np.linspace(-(length/2), (length/2), num=resolution)
    x = np.array([x_coords,]*len(y_coords))
    y = np.array([y_coords,]*len(x_coords)).transpose()    
    #initialise empty dictionary for centres and empty total_feature_array
    centres={}
    total_feature_array = np.zeros((len(y_coords),len(x_coords)))    
    #loop over each feature, check if its a square of circle first
    for feature in features:
        if feature[0]=='B':
            possible_centres={}
            #loop over x and y coordinates of the array, checking if they are part of the feature
            for i in range(len(x_coords)):
                for j in range(len(y_coords)):
                    #point is part of square feature if it is less than half the length of the square away
                    #from the centre in both x and y directions
                    if (abs(x[(j,i)]-feature[3])<=(feature[1]/2)) and (abs(y[(j,i)]-feature[4])<=(feature[2]/2)):
                        #each point that is a part of the square becomes a possible centre with a given distance from the 
                        #absolute centre and the point becomes 1 on the total_feature_array
                        possible_centres[(i,j)]=np.sqrt(np.square(x[(j,i)]-feature[3]) + np.square(y[(j,i)]-feature[4]))
                        total_feature_array[(j,i)]=1
            #calculate which point in the feature is closest to the absolute centre and adds this to
            #centres dictionary
            centres[min(possible_centres, key=possible_centres.get)]=feature
            print(features.index(feature))
        if feature[0]=='R':
            possible_centres={}
            #loop over x and y coordinates of the array, checking if they are part of the feature
            for i in range(len(x_coords)):
                for j in range(len(y_coords)):
                    #point is part of a circle feature if the euclidian distance from the point to the circle centre
                    #is smaller than the radius of the circle
                    if np.sqrt(np.square(x[(j,i)]-feature[2]) + np.square(y[(j,i)]-feature[3]))<=feature[1]:
                        #each point that is a part of the square becomes a possible centre with a given distance from the 
                        #absolute centre and the point becomes 1 on the total_feature_array
                        total_feature_array[(j,i)]=1
                        possible_centres[(i,j)]=np.sqrt(np.square(x[(j,i)]-feature[2]) + np.square(y[(j,i)]-feature[3]))
            #calculate which point in the feature is closest to the absolute centre and adds this to
            #centres dictionary
            centres[min(possible_centres, key=possible_centres.get)]=feature
            print(features.index(feature))            
    #calculate time passed since starting function
    final_time = time.time() - t
    print('time for discretization is', final_time)       
    return total_feature_array, centres


def get_feature_map(centre, centres, total_feature_array, dimensions):
    '''Can create an array for a feature as it is needed for the algorithm.'''
    length = dimensions[0]
    resolution = dimensions[1]    
    #create empty arrays
    x_coords = np.linspace(-(length/2), (length/2), num=resolution)
    y_coords = np.linspace(-(length/2), (length/2), num=resolution)
    x = np.array([x_coords,]*len(y_coords))
    y = np.array([y_coords,]*len(x_coords)).transpose()    
    feature_array = np.full_like(total_feature_array, 0)
    #create feature arrays
    feature = centres[centre]
    if feature[0]=='B':
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                if (abs(x[(j,i)]-feature[3])<=(feature[1]/2)) and (abs(y[(j,i)]-feature[4])<=(feature[2]/2)):
                    feature_array[(j,i)]=1
    if feature[0]=='R':
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                if np.sqrt(np.square(x[(j,i)]-feature[2]) + np.square(y[(j,i)]-feature[3]))<=feature[1]:
                    feature_array[(j,i)]=1
    return feature_array


def order_kNN_s(centres):
    '''kNN style ordering of starts and goals, shortest first.'''
    starts=[]
    goals=[]
    start_to_goal={}
    for centre in centres:
        if centres[centre][0]=='B':
            starts.append(centre)
        if centres[centre][0]=='R':
            goals.append(centre)
    if len(starts)==len(goals):
        print('Number of starts and goals are the same')
    else:
        return print('Error: number of starts and goals not equal')
    for start in starts:
        goal_distances = []
        for goal in goals:
            distance = pfa.heuristic_m(start, goal)
            goal_distances.append([goal, distance])
        goal_distances.sort(key=lambda x: x[1])
        start_to_goal[start]=goal_distances
    start_to_goal=dict(sorted(start_to_goal.items(), key=lambda item: item[1][0][1]))
            
    return start_to_goal

def order_heuristic_split(centres):
    '''vertical heuristic style ordering of starts and goals. goes per column from left to right.
    splits up into two groups: up and down'''
    starts=[]
    goals=[]
    start_to_goal_down={}
    start_to_goal_up={}
    #first split centres into starts (squares) and goals (circles)
    for centre in centres:
        if centres[centre][0]=='B':
            starts.append(centre)
        if centres[centre][0]=='R':
            goals.append(centre)
    #check whether the number of starts and goals are the same
    if len(starts)==len(goals):
        print('Number of starts and goals are the same')
    else:
        return print('Error: number of starts and goals not equal')
    #calculate the distance to each circle for every square
    for start in starts:
        goal_distances = []
        split = max(centres)[1]*0.2
        for goal in goals: 
            #for downwards group, the distance is the horizontal and then vertical distance
            if start[1]>=split:
                distance = [abs(start[0]-goal[0]),abs(start[1]-goal[1])]
            #for upwards group, the distance is the octile distance
            if start[1]<split:
                distance = pfa.heuristic_o(start, goal)
            goal_distances.append([goal, distance]) 
        #sort goal distances and add to dictionaries
        if start[1]>=split:
            goal_distances.sort(key=lambda x: (x[1][0],x[1][1]))
            start_to_goal_down[start]=goal_distances
        if start[1]<split:
            goal_distances.sort(key=lambda x: x[1])
            start_to_goal_up[start]=goal_distances
    #sort dictionaries based on defined order
    sorting_number = (max(start_to_goal_down)[0]-min(start_to_goal_down)[0])//2 + min(start_to_goal_down)[0]
    #for downwards group order per column starting in the middle and from down to up per column
    start_to_goal_down=dict(sorted(start_to_goal_down.items(), key=lambda key: (abs(key[0][0]-sorting_number),-key[0][1])))
    #for upwards group per row starting from the middle, and from down to up per row
    start_to_goal_up=dict(sorted(start_to_goal_up.items(), key=lambda key: (-key[0][1],-abs((key[0][0])-sorting_number))))
    #return the two dictionaries to be used for a* algorithm     
    return start_to_goal_down, start_to_goal_up



#execution functions    
total_feature_array, centres = discretize(features_final_mirror,dimensions)
start_to_goal_down, start_to_goal_up = order_heuristic_split(centres)
start_to_goal = order_kNN_s(centres)













