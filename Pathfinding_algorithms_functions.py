# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:56:53 2021

@author: Gebruiker
"""

import heapq
import math
from typing import Dict, List, Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')
Location = TypeVar('Location')
  
class PriorityQueue:
    '''Create a priority queue: a list of tuples with (priority,item). Function empty() returns 
    True if the queue is empty and False if it is not. Function put() adds an element with a
    given priority to the queue. Function get() removes the item with the lowest cost (highest
    priority) and returns the item itself.'''
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]
    
def heuristic_m(a,b) -> float:
    '''Calculates the manhattan distance between two points a and b, and returns
    this distance.'''
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic_e(a,b) -> float:
    '''Calculates the euclidian distance between two points a and b, and returns 
    this distance.'''
    (x1, y1) = a
    (x2, y2) = b
    euclidian = (abs(x1-x2)^2) + (abs(y1-y2)^2)
    return math.sqrt(euclidian)

def heuristic_o(a,b) -> float:
    '''Calculates the octile distance between two points a and b, and returns 
    this distance.'''
    (x1, y1) = a
    (x2, y2) = b
    dx = abs(x1-x2)
    dy = abs(y1-y2)
    return math.sqrt(2)*min(dx,dy)+abs(dx-dy)

def reconstruct_path(came_from, start, goal):
    '''Reconstructs the path from the result of a pathfinding algorithm. It works its way
    backwards from the goal given the came_from dictionary.'''
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

        
def cost(from_location, to_location):
    '''Calculates the cost of moving from one node to another. It can either be a horizontal, vertical 
    or diagonal move. Horizontal and vertical both have cost 1, where as a diagonal move has cost 
    sqrt(2).'''
    difference = abs(from_location[1]-to_location[1])+abs(from_location[0]-to_location[0])
    if difference==1:
        return 1
    if difference==2:
        return math.sqrt(2)
    
def neighbors(array, location):
    '''Finds all 8 neighbors of a given node (including diagonal neighbors). It then checks
    if these neighbors are in the bounds of the array and are not considered obstacles. A list
    of these neighbors is returned.'''
    (x, y) = location
    neighbors= [(x+1, y), (x-1, y), (x, y-1), (x, y+1), (x+1, y+1), (x+1, y-1), (x-1, y-1), (x-1, y+1)] 
    neighbors_filtered=[]
    for i in neighbors:
        if (0 <= i[1] < array.shape[0]) and (0 <= i[0] < array.shape[1]):
            if (not (array[(i[1],i[0])]==1)):
                neighbors_filtered.append(i)
    return neighbors_filtered   

def neighbors_restricted(array, location):
    '''Finds the 4 neighbors of a given node (so not including diagonal neighbors). It then checks
    if these neighbors are in the bounds of the array and are not considered obstacles. A list
    of these neighbors is returned.''' 
    (x, y) = location
    neighbors= [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] 
    neighbors_filtered=[]
    for i in neighbors:
        if (0 <= i[1] < array.shape[0]) and (0 <= i[0] < array.shape[1]):
            if (not (array[(i[1],i[0])]==1)):
                neighbors_filtered.append(i)
    return neighbors_filtered 

def path_length(paths):
    '''Determines the total path length from all completed paths in array points
    and returns this length.'''
    path_lengths =[]
    #loop over all paths 
    for path in paths:
        path_length=0
        #loop over all points in path
        for i in range(len(path)-1):
            #for horizontal and vertical movement length 1
            if (abs(path[i][0]-path[i+1][0])+abs(path[i][1]-path[i+1][1]))==1:
                path_length += 1
            #for diagonal movement length sqrt(2)
            if (abs(path[i][0]-path[i+1][0])+abs(path[i][1]-path[i+1][1]))==2:
                path_length += math.sqrt(2)
        path_lengths.append(path_length)
    total_path_length = sum(path_lengths)
    return total_path_length

def path_length_single(path):
    '''Determines the path length of a single path in array points and returns
    this length.'''
    path_length=0
    #loop over all points in path
    for i in range(len(path)-1):
        #for horizontal and vertical movement length 1
        if (abs(path[i][0]-path[i+1][0])+abs(path[i][1]-path[i+1][1]))==1:
            path_length += 1
        #for diagonal movement length sqrt(2)
        if (abs(path[i][0]-path[i+1][0])+abs(path[i][1]-path[i+1][1]))==2:
            path_length += math.sqrt(2)
    return path_length 
    