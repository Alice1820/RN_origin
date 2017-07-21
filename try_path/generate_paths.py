#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:11:20 2017

@author: xifan
"""
import numpy as np

coor_to_index = np.zeros((7, 7))
for n in range(7 * 7):
    x = int(n / 7)
    y = int(n % 7)
    coor_to_index[x, y] = n
#print (coor_to_index)

eight_neighbor = [(0, 1), (1, 0), (1, 1)]
unit_1 = [[(0, 1)], [(1, 0)], [(1, 1)]]

def generate_path_unit(length):
    if length == 1:
        return unit_1
    else:
        unit = []
#        unit.append()
        unit_0 = generate_path_unit(length = length-1)
        for path in unit_0:
            print (path)
            for x, y in eight_neighbor:
#                print (path)
                x_new = path[-1][0] + x
                y_new = path[-1][1] + y
                path_new = []
                path_new += (path)
                path_new.append((x_new, y_new))
#                print (path_new)
                unit.append(path_new)
        
        return unit
        
def generate_paths(height, width, length):
    
    unit = generate_path_unit(length = length)
    PATHS = []
    for x in range(height):
        for y in range(width):
            for path in unit:
#                print (path)
                x_end = path[-1][0] + x
                y_end = path[-1][1] + y
                if x_end >= 0 and x_end <= height-1 and y_end >= 0 and y_end <= width-1:
                    path_new = []
                    path_new.append((x, y))
                    for x_move, y_move in path:
                        path_new.append((x + x_move, y + y_move))
                    PATHS.append(path_new)
                    path_new.reverse()
                    PATHS.append(path_new)
    print (len(PATHS))
    filename = "paths_" + str(height) + '_' + str(width) + '_' + str(length) + '.txt'
    print (filename)
    f = open('paths.txt', 'w')
    f.write(str(PATHS))
    f.close()
    
    return PATHS
    
#print (generate_path_unit(length = 2))
PATHS = generate_paths(7, 7, 3)