# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

# def sin(theta):
#     "Taylor Approx of Sin"
#     return (theta - pow(theta, 3)/6 + pow(theta, 5)/120 - pow(theta, 7)/5040)

# def cos(theta):
#     "Taylor Approx of Cos"
#     return (1 - pow(theta, 2)/2 + pow(theta, 4)/24 - pow(theta, 6)/720)

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    # end_x = start_x + math.floor(np.cos(math.radians(angle))*length)
    # end_y = start_y - math.floor(np.sin(math.radians(angle))*length)
    return (start[0] + np.cos(math.radians(angle))*length, start[1] - np.sin(math.radians(angle))*length)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    for arm in armPosDist:
        for obj in objects:
            arm_start = arm[0] #(x0, y0)
            arm_end = arm[1] 
            # arm_vector = np.array([arm_end[0] - arm_start[0], arm_end[1] - arm_start[1]])
            # goal_vector = np.array([obj[0] - arm_start[0], obj[1] - arm_start[1]])
            
            Dx = arm_end[0] - arm_start[0]
            Dy = arm_end[1] - arm_start[1]
            
            orth = Dx**2 + Dy**2
            
            dot = ((obj[0] - arm_start[0]) * Dx + (obj[1] - arm_start[1]) * Dy)/float(orth)
            
            #line segment piecewise function
            if dot < 0:
                dot  = 0
            elif dot > 1:
                dot = 1
            
            # x = arm_start[0] + dot * Dx
            # y = arm_start[1] + dot * Dy
            nx = arm_start[0] + dot * Dx - obj[0]
            ny = arm_start[1] + dot * Dy - obj[1]
            distance_squared = (nx**2 + ny**2)
            
            r = obj[2] + arm[2] #effective radius = object radius + padding distance
            if isGoal:
                r = obj[2]
            if r**2 >= distance_squared:
                return True
    return False

def d_squared(a, b):
    return pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2)

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tick touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tip touches any goal. False if not.
    """
    for goal in goals:
        cur_goal = (goal[0], goal[1])
        radius = goal[2]
        if d_squared(armEnd, cur_goal) <= pow(radius, 2):
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    width = window[0]
    height = window[1]
    for arm in armPos:
        start = arm[0]
        end = arm[1]
        if start[0] < 0 or start[0] > width or start[1] < 0 or start[1] > height:
            return False
        if end[0] < 0 or end[0] > width or end[1] < 0 or end[1] > height:
            return False
    
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTickTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
