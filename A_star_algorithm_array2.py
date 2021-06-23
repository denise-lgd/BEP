# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:12:26 2021

@author: Gebruiker
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:34:06 2021

@author: Gebruiker
"""
import Pathfinding_algorithms_functions as pfa
import numpy as np
import matplotlib.pyplot as plt
import math
import time


    
def a_star_search(array, start, goal):
    '''Performs a* search algorithm. Function neighbors() find all neighbors including diagonals,
    and function neighbors_restricted() finds only the 4 neighbors not including diagonals.'''
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in neighbors_restricted(array, current):
            new_cost = cost_so_far[current] + cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + pfa.heuristic_o(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def optimize(path):
    '''Rounds corners of a path to shorten the path length.'''
    i=1
    removed_points =[]
    #there must be 3 points avaiable to evaluate whether this is a corner, so i cannot be 
    #larger than len(path)-2
    while i<=len(path)-2:
        #use the fact that a corner is a horizontal followed by a vertical move
        if (abs(path[i-1][0]-path[i+1][0])==1 and abs(path[i-1][1]-path[i+1][1])==1):
            #remove the middle point of the corner to create diagonal line
            removed_points.append(path[i])
            #if corner found, move two points up to avoid diagonal lines crossing eachother
            i=i+2
        else:
            #if no corner found, move one point up
            i=i+1
    new_path = []
    #form new path without the removed points
    for point in path:
        if point not in removed_points:
            new_path.append(point)
    return new_path


def execute_astar(total_feature_array, start_to_goal, centres, dimensions):
    '''Executes the A* algorithm for a given feature array, start_to_goal dictionary, centres and 
    dimensions. Used for the kNN-style heuristic.'''
    #make a copy of the total feature array to avoid altering the original array
    total_feature_array_copy = total_feature_array.copy()
    #initialise empty lists/numbers for taken goals, paths and the total time
    taken_goals=[]
    paths = []
    total_time = 0
    #create vertical line along symmetry axis so it doesn't get crossed by paths
    total_feature_array_copy[:,(np.shape(total_feature_array)[0]//2)]=1
    #loop over starts, count keeps track of which iteration we are on
    for count,starts in enumerate(start_to_goal):
        #use try/except structure so it can continue if an error occurs
        try:
            #initialise time, index, start and goal
            t=time.time()
            index=0
            start = starts
            goal = 0
            #determine most preferable goal for start which is not already taken
            while goal==0:
                if start_to_goal[starts][index][0] in taken_goals:
                    index +=1
                else: 
                    goal = start_to_goal[starts][index][0]
            #once goal has been selected add to taken goals
            taken_goals.append(goal)
            #collect feature arrays and perform a* algorithm to find path
            start_array = get_feature_map(start, centres, total_feature_array, dimensions)
            goal_array = get_feature_map(goal, centres, total_feature_array, dimensions)
            array = total_feature_array_copy - (start_array + goal_array)  
            came_from, cost_so_far = a_star_search(array, start, goal)
            initial_path = reconstruct_path(came_from, start, goal)
            path = optimize(initial_path)
            #make the found path 1's on the total feature array copy 
            for point in path:
                total_feature_array_copy[(point[1],point[0])]=1
            #calculate the time passed and print relevant information
            time_passed = time.time() - t
            total_time += time_passed
            print('iteration',count,'from',start,'to',goal,'completed in',time_passed,'seconds')
            paths.append(path)
        #if path cannot be completed a key error occurs when trying to use reconstruct_path()
        except KeyError:
            #print information and continue onto the next start
            print('path from', start, 'to', goal, 'cannot be found')
            continue
    #calculate total path length
    total_path_length = path_length(paths)
    return paths, total_path_length, total_time


def execute_astar_split(total_feature_array, start_to_goal_down, start_to_goal_up, centres, dimensions):
    '''Executes the A* algorithm for the split heuristic given a total feature array, start_to_goal_down and start_to_goal_up
    dictionaries, centres and dimensions.'''
    #make a copy of the total feature array to avoid altering the original array
    total_feature_array_copy = total_feature_array.copy()
    #initialise empty lists/numbers for taken goals, paths and the total time
    taken_goals=[]
    paths_down = {}
    paths_up = {}
    total_time = 0
    #create vertical line along symmetry axis so it doesn't get crossed by paths
    total_feature_array_copy[:,(np.shape(total_feature_array)[0]//2)]=1
    #manual alteration line
    total_feature_array_copy[997:1446,911]=1
    #loop over starts in downwards group, count keeps track of which iteration we are on
    i=0
    starts = list(start_to_goal_down.keys())
    #keep track of which iterations have been completed
    completed_down = []
    while i < len(start_to_goal_down) and i>=0:
        if i in completed_down:
            i+=1
            continue
        else:
            #use try/except structure so it can continue if an error occurs
            try:
                #initialise time, index, start and goal
                t=time.time()
                index=0
                start = starts[i]
                goal = 0
                #determine most preferable goal for start which is not already taken
                while goal==0:
                    if start_to_goal_down[start][index][0] in taken_goals:
                        index +=1
                    else: 
                        goal = start_to_goal_down[start][index][0]                
                #collect feature arrays and perform a* algorithm to find path
                start_array = get_feature_map(start, centres, total_feature_array, dimensions)
                goal_array = get_feature_map(goal, centres, total_feature_array, dimensions)
                array = total_feature_array_copy - (start_array + goal_array)
                came_from, cost_so_far = a_star_search(array, start, goal)
                initial_path = reconstruct_path(came_from, start, goal)
                path = optimize(initial_path)
                #once goal has been selected add to taken goals
                taken_goals.append(goal)
                #make the found path 1's on the total feature array copy
                for point in path:
                    total_feature_array_copy[(point[1],point[0])]=1
                #calculate time passed and print relevant information
                time_passed = time.time() - t
                total_time += time_passed
                print('iteration',i,'from',start,'to',goal,'completed in',time_passed,'seconds')
                paths_down[i]=path
                completed_down.append(i)
                i+=1
            #if path cannot be completed a key error occurs when trying to use reconstruct_path()
            except KeyError:
                print('path from', start, 'to', goal, 'cannot be found so backtrack')
                j=i
                completed_path = False
                while completed_path == False and j>=0:
                    j=j-1
                    #remove previous iteration
                    for point in paths_down[j]:
                        total_feature_array_copy[(point[1],point[0])]=0
                    taken_goals.remove(paths_down[j][-1])
                    paths_down.pop(j)
                    completed_down.remove(j)
                    try:
                        t=time.time()
                        #perform a* algorithm again to find path                   
                        array = total_feature_array_copy - (start_array + goal_array)
                        came_from, cost_so_far = a_star_search(array, start, goal)
                        initial_path = reconstruct_path(came_from, start, goal)
                        path = optimize(initial_path)
                        #once goal has been selected add to taken goals
                        taken_goals.append(goal)
                        #make the found path 1's on the total feature array copy
                        for point in path:
                            total_feature_array_copy[(point[1],point[0])]=1
                        #calculate time passed and print relevant information
                        time_passed = time.time() - t
                        total_time += time_passed
                        print('iteration',i,'from',start,'to',goal,'completed in',time_passed,'seconds')
                        paths_down[i]=path
                        completed_path = True
                        completed_down.append(i)
                        i=0                    
                    except KeyError:
                        #if backtracking did not work remove another iteration
                        print('remove another path')
                        #if all paths have been removed exit while loop
                        if j==0:
                            print('backtracking did not work')
                            break
                        else:
                            continue
    #loop over starts in upwards group, count keeps track of which iteration we are on
    i=0
    starts = list(start_to_goal_up.keys())
    #keep track of which iterations have been completed
    completed_up = []
    while i < len(start_to_goal_up) and i>=0:
        if i in completed_up:
            i=i+1
            continue
        else:
            #use try/except structure so it can continue if an error occurs
            try:
                print(completed_up)
                #initialise time, index, start and goal
                t=time.time()
                index=0
                start = starts[i]
                goal = 0
                #determine most preferable goal for start which is not already taken
                while goal==0:
                    if start_to_goal_up[start][index][0] in taken_goals:
                        index +=1
                    else: 
                        goal = start_to_goal_up[start][index][0]
                #collect feature arrays and perform a* algorithm to find path
                start_array = get_feature_map(start, centres, total_feature_array, dimensions)
                goal_array = get_feature_map(goal, centres, total_feature_array, dimensions)
                array = total_feature_array_copy - (start_array + goal_array)
                came_from, cost_so_far = a_star_search(array, start, goal)
                initial_path = reconstruct_path(came_from, start, goal)
                path = optimize(initial_path)
                #once goal has been selected add to taken goals
                taken_goals.append(goal)
                #make the found path 1's on the total feature array copy
                for point in path:
                    total_feature_array_copy[(point[1],point[0])]=1
                #calculate time passed and print relevant information
                time_passed = time.time() - t
                total_time += time_passed
                print('iteration',i,'from',start,'to',goal,'completed in',time_passed,'seconds')
                paths_up[i]=path
                completed_up.append(i)
                i=+1
            #if path cannot be completed a key error occurs when trying to use reconstruct_path()
            except KeyError:
                print('path from', start, 'to', goal, 'cannot be found so backtrack')                
                completed_path = False
                j=i
                while completed_path == False and j>=0:
                    j=j-1
                    #remove previous iteration
                    for point in paths_up[j]:
                        total_feature_array_copy[(point[1],point[0])]=0
                    taken_goals.remove(paths_up[j][-1])
                    paths_up.pop(j)
                    completed_up.remove(j)
                    try:
                        t=time.time()                        
                        #collect feature arrays and perform a* algorithm to find path
                        array = total_feature_array_copy - (start_array + goal_array)
                        came_from, cost_so_far = a_star_search(array, start, goal)
                        initial_path = reconstruct_path(came_from, start, goal)
                        path = optimize(initial_path)
                        taken_goals.append(goal)
                        #make the found path 1's on the total feature array copy
                        for point in path:
                            total_feature_array_copy[(point[1],point[0])]=1
                        #calculate time passed and print relevant information
                        time_passed = time.time() - t
                        total_time += time_passed
                        print('iteration',i,'from',start,'to',goal,'completed in',time_passed,'seconds')
                        paths_up[i]=path
                        completed_path = True
                        completed_up.append(i)
                        i=0 
                    except KeyError:
                        print('remove another path')
                        if j==0:
                            print('backtracking did not work')
                            break
                        else:
                            continue
    #return the final order of the paths  
    completed_all = completed_down + completed_up
    #calculate total path length 
    paths = []
    for path in paths_down:
        paths.append(paths_down[path])
    for path in paths_up:
        paths.append(paths_up[path]) 
    total_path_length = path_length(paths)   
    return paths, total_path_length, total_time, completed_all


def mirror_plot(total_feature_array, paths, dimensions):
    '''Takes half of the layout and its paths as input, and will mirror this in the vertical middle 
    line to create the full layout. It also transforms the result and places it on the wafer''' 
    #make a copy of the total feature array to avoid altering the original array
    total_feature_array_copy = total_feature_array.copy()
    x_length = np.shape(total_feature_array)[1]
    y_length = np.shape(total_feature_array)[0]
    #loop through all points. Any point which is a 1 will be mirrored.    
    for x in range(x_length):
        for y in range(y_length):
            if total_feature_array[(y,x)]==1:
                total_feature_array_copy[(y,-x)]=1
    #transformation to make sure layout is in the middle of the wafer
    transformation = 125
    #make another different copy for the transformed array and fill with 0's
    total_feature_array_transformed = total_feature_array_copy.copy()
    total_feature_array_transformed.fill(0)
    #any 1 will be transformed
    for x in range(x_length):
        for y in range(y_length):
            if total_feature_array_copy[(y,x)]==1:
                total_feature_array_transformed[(y+transformation,x)]=1
    #create a mirrored path for each path
    all_paths = []
    #loop over paths created
    for path in paths:
        mirrored_path = []
        #loop over each point in path
        for point in path:
            distance_to_middle = abs(point[0]-(x_length/2))
            mirrored_path.append(((x_length/2)+distance_to_middle,point[1]))
        #all_paths will contain both original and mirrored paths
        all_paths.append(path)
        all_paths.append(mirrored_path)
    #visualize wafer on total_feature_array_transformed
    radius_wafer = dimensions[2]   
    for x in range(x_length):
        for y in range(y_length):
            #check whether a node is part of the wafer
            if np.sqrt(np.square(x-(x_length/2)) + np.square(y-(y_length/2)))>=((radius_wafer)/dimensions[0])*dimensions[1]:
                total_feature_array_transformed[(y,x)]=1    
    #visualize transformed total feature array and paths
    fig, ax = plt.subplots(figsize=(15,15))   
    ax.imshow(total_feature_array_transformed, cmap='binary')
    for path in all_paths:
        x=[i[0] for i in path]
        y=[(i[1]+transformation) for i in path]
        ax.plot(x,y,  color='blue' ,linestyle='-')    
    plt.show()    
    return total_feature_array_transformed


#execution split heuristic
paths, total_path_length, total_time, order = execute_astar_split(total_feature_array, start_to_goal_down, start_to_goal_up, centres, dimensions)
print('The total path length is', total_path_length, 'for', len(paths), 'paths completed')
print('The total time is',total_time,'seconds')
feature_array_transformed = mirror_plot(total_feature_array, paths, dimensions) 


#execution kNN
#paths, total_path_length, total_time = execute_astar(total_feature_array, start_to_goal, centres, dimensions)  
