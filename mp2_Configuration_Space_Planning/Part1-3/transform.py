
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *


def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    
    alpha_start, beta_start = arm.getArmAngle()
    (alpha_min, alpha_max), (beta_min, beta_max) = arm.getArmLimit()
    
    # number of positions in []
    alpha_positions = int((alpha_max - alpha_min)/granularity + 1)
    beta_positions = int((beta_max - beta_min)/granularity + 1)
    
    # initialize maze 2D list
    maze = [[SPACE_CHAR for beta_step in range(beta_positions)] for alpha_step in range(alpha_positions)]

    for alpha in range(alpha_min, alpha_max, granularity):
        alpha_step = int((alpha- alpha_min)/granularity)
        beta_optimization = False
        for beta in range(beta_min, beta_max, granularity):
            beta_step = int((beta - beta_min)/granularity)            
            if beta_optimization:
                maze[alpha_step][beta_step] = WALL_CHAR
                continue
            
            arm.setArmAngle((alpha, beta))
            armPos = arm.getArmPos()
            armPosDist = arm.getArmPosDist()
            armEnd = armPos[-1][-1]
            
            if doesArmTouchObjects(armPosDist[:-1], obstacles, ):
                beta_optimization = True
                maze[alpha_step][beta_step] = WALL_CHAR
            elif doesArmTouchObjects(armPosDist, obstacles, ):
                maze[alpha_step][beta_step] = WALL_CHAR
            elif isArmWithinWindow(armPos, window) == False:
                maze[alpha_step][beta_step] = WALL_CHAR
            elif doesArmTipTouchGoals(armEnd, goals):
                maze[alpha_step][beta_step] = OBJECTIVE_CHAR  
    
    start_alpha_step, start_beta_step = angleToIdx([alpha_start, beta_start], [alpha_min, beta_min], granularity)    
    maze[start_alpha_step][start_beta_step] = START_CHAR
    
    return Maze(maze, [alpha_min, beta_min], granularity)
            
            