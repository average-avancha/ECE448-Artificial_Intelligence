# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

import heapq

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def sanity_check(maze, path):
    """
    Runs check functions for part 0 of the assignment.

    @param maze: The maze to execute the search on.
    @param path: a list of tuples containing the coordinates of each state in the computed path

    @return bool: whether or not the path pass the sanity check
    """
    # TODO: Write your code here
    return False


def outPath(previous, start, end):
    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = previous[cur]
    return path

def bfs(maze):
    dots = maze.getObjectives()
    
    while len(dots) > 0:
        start = maze.getStart()
        path = [start]
        previous = {}
        discovered = {}
        discovered[start] = True
        
        q = []
        q.append(start)
        end = dots[0]
        while len(q) > 0:
            cur = q.pop(0)
            if(cur == end):
                #return the path
                out = outPath(previous, start, end)
                out.reverse()
                path = path + out
                dots.pop(0)
                break
            
            neighbors = maze.getNeighbors(cur[0], cur[1])
            for point in neighbors:
                if point not in discovered:
                    #print("add: " + point)
                    q.append(point)
                    previous[point] = cur
                    discovered[point] = True       
    return path
    
    """
    Runs BFS for part 1 of the assignment.
    
    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
def ConstructPath(previous, start, end):
    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = previous[cur]
    path.reverse()
    return path

def astar(maze):
    dots = maze.getObjectives()
    start = maze.getStart()
    path = [start]
    visited = {}
    end = dots[0]
    
    openSet = [start]
    cameFrom = {}
    
    #g[cur] = lowest cost from start to cur
    g = {}
    g[start] = 0
    
    #f[cur] = g[cur] + h[cur] --> estimate of total cost
    f = {}
    f[start] = hman(start, end)
    
    while len(openSet) > 0:
        cur = openSet[0]
        visited[cur] = True
        if cur == end:
            #end reached
            dots.pop(0)
            path = path + ConstructPath(cameFrom, start, end)
            break
        openSet.pop(0)
        neighbors = maze.getNeighbors(cur[0], cur[1])
        for neighbor in neighbors:
            g[neighbor] = g[cur] + 1
            f[neighbor] = g[neighbor] + hman(neighbor, end)
            if neighbor not in visited.keys():
                visited[neighbor] = False
            if neighbor not in openSet and visited[neighbor] is False:
                cameFrom[neighbor] = cur
                if len(openSet) == 0:
                    openSet.insert(0, neighbor)
                else:
                    for i in range(len(openSet)):
                        if f[neighbor] <= f[openSet[i]]:
                            openSet.insert(i, neighbor)
                            break
                        if (i + 1 == len(openSet)) and (f[neighbor] > f[openSet[i]]):
                            openSet.insert(i + 1, neighbor)
                        
    #return no path
    return path
    
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # return []

def hman(point, end):
    #manhattan distance from point to end
    return abs(point[0] - end[0]) + abs(point[1] - end[1])

def hastar(maze, start, end):
    path = [start]
    visited = {}
    
    openSet = [start]
    cameFrom = {}
    
    #g[cur] = lowest cost from start to cur
    g = {}
    g[start] = 0
    
    #f[cur] = g[cur] + h[cur] --> estimate of total cost
    f = {}
    f[start] = hman(start, end)
    
    while len(openSet) > 0:
        cur = openSet[0]
        visited[cur] = True
        if cur == end:
            #end reached
            path = path + ConstructPath(cameFrom, start, end)
            break
        openSet.pop(0)
        neighbors = maze.getNeighbors(cur[0], cur[1])
        for neighbor in neighbors:
            g[neighbor] = g[cur] + 1
            f[neighbor] = g[neighbor] + hman(neighbor, end) 
            if neighbor not in visited.keys():
                visited[neighbor] = False
            if neighbor not in openSet and visited[neighbor] is False:
                cameFrom[neighbor] = cur
                if len(openSet) == 0:
                    openSet.insert(0, neighbor)
                else:
                    for i in range(len(openSet)):
                        if f[neighbor] <= f[openSet[i]]:
                            openSet.insert(i, neighbor)
                            break
                        if (i + 1 == len(openSet)) and (f[neighbor] > f[openSet[i]]):
                            openSet.insert(i + 1, neighbor)
                        
    return len(path)

def spanning_man(maze, goals, start, E):
    (nearest, cost) = get_goal_man(maze, start, goals, E)
    distance = cost
    if len(goals) == 0:
        return distance
    
    V = [nearest] #visited set
    U = list(goals[:]) #unvisited set
    U.remove(nearest)
    
    if len(U) == 0:
        return distance
    
    while len(U) != 0:
        mincost = float('inf')
        min_flag = False
        for u in U:
            for v in V:
                if u != v:
                    if (u, v) not in E.keys():
                        cost = hman(u, v)
                        E[(u, v)] = cost
                        E[(v, u)] = cost
                    cost = E[(u, v)]
                    if cost < mincost:
                        min_flag = True
                        mincost = cost
                        minnode = u    
        # if unvisited node not in Visited list then add to list
        if min_flag == True: # minset flag to fix bug
            V.append(minnode)
            U.remove(minnode)
            distance += mincost
            min_flag = False
        
    #return optimium total distance
    return distance

def spanning_astar(maze, goals, start, E):
    (nearest, cost) = get_goal(maze, start, goals, E)
    distance = cost
    if len(goals) == 0:
        return distance
    
    V = [nearest] #visited set
    U = list(goals[:]) #unvisited set
    U.remove(nearest)
    
    if len(U) == 0:
        return distance
    
    
    while len(U) != 0:
        mincost = float('inf')
        min_flag = False
        for u in U:
            for v in V:
                if u != v:
                    if (u, v) not in E.keys():
                        cost = hastar(maze, u, v)
                        E[(u, v)] = cost
                        E[(v, u)] = cost
                    cost = E[(u, v)]
                    if cost < mincost:
                        min_flag = True
                        mincost = cost
                        minnode = u    
        # if unvisited node not in Visited list then add to list
        if min_flag == True: # minset flag to fix bug
            V.append(minnode)
            U.remove(minnode)
            distance += mincost
            min_flag = False
        
    #return optimium total distance
    return distance

def get_goal(maze, start, goals, E):
    mincost = float('inf')
    closest = tuple()
    for goal in goals:
        if (start, goal) not in E.keys() or (goal, start) not in E.keys():
            cost = hastar(maze, start, goal)
            E[(start, goal)] = cost
            E[(goal, start)] = cost
        cost = E[(start, goal)]
        if cost < mincost:
            mincost = cost
            closest = goal
    return (closest, mincost)

def get_goal_man(maze, start, goals, E):
    mincost = float('inf')
    closest = tuple()
    for goal in goals:
        if (start, goal) not in E.keys() or (goal, start) not in E.keys():
            cost = hman(start, goal)
            E[(start, goal)] = cost
            E[(goal, start)] = cost
        cost = E[(start, goal)]
        if cost < mincost:
            mincost = cost
            closest = goal
    return (closest, mincost)

def ConstructPath_State(previous_dict, state):
    path = []
    cur_state = state
    while previous_dict[cur_state] != "start":
        path.append(cur_state[2])
        cur_state = previous_dict[cur_state]
    path.reverse()
    return path

def astar_corner(maze):
    path = [maze.getStart()]
    
    #E[pointA, pointB] = actual min distance from point A to B
    E = {}
    
    #g[(cur, goals)] = lowest cost from start to cur
    g = {}
    g[(maze.getStart(), tuple(maze.getObjectives()))] = 0
    
    #h[(cur, goals)] = approx cost to end from cur
    h = {}
    h[(maze.getStart(), tuple(maze.getObjectives()))] = spanning_man(maze, maze.getObjectives(), maze.getStart(), E)
    
    #f[(cur, goals)] = g[cur] + h[cur] --> estimate of total cost
    # f = {}
    # f[(maze.getStart(), maze.getObjectives)] =
    
    # state = (f, g, cur, remaining goals)
    start = (h[(maze.getStart(), tuple(maze.getObjectives()))], 0, maze.getStart(), tuple(maze.getObjectives()))
    
    prev = {}
    prev[start] = "start"
    
    priorityq = []
    #heapq.heapify(priorityq)
    heapq.heappush(priorityq, start)
    
    while len(priorityq) > 0:
        state = heapq.heappop(priorityq)
        cur = state[2]
        goals = state[3]
        
        if cur in goals:
            #pop cur from goals
            old_goals = goals
            new_goals = []
            for goal in goals:
                if goal != cur:
                    new_goals.append(goal)
            goals = tuple(new_goals)
            
            g[(cur, goals)] = g[(cur, old_goals)]
            h[(cur, goals)] = h[(cur, old_goals)]
            prev[(g[(cur, goals)]+h[(cur, goals)], g[(cur, goals)], cur, goals)] = prev[(g[(cur, old_goals)]+h[(cur, old_goals)], g[(cur, old_goals)], cur, old_goals)]
        
        if len(goals) == 0:
            #end reached
            path = path + ConstructPath_State(prev, state)
            break
        # print("Cur: ", cur)
        neighbors = maze.getNeighbors(cur[0], cur[1])
        for neighbor in neighbors:
            # update g[neighbor, goals]
            if (neighbor, goals) not in g.keys():
                g[(neighbor, goals)] = g[(cur, goals)] + 1
            elif (g[(cur, goals)] + 1) < g[(neighbor, goals)]:
                g.pop((neighbor, goals), None)
                g[(neighbor, goals)] = g[(cur, goals)] + 1
            # update h[neighbor, goals]
            if (neighbor, goals) not in h.keys():
                h[(neighbor, goals)] = spanning_man(maze, goals, cur, E)
            
            neighbor_state = (h[(neighbor, goals)] + g[(neighbor, goals)], g[(neighbor, goals)], neighbor, goals)
            
            if neighbor_state not in priorityq and neighbor_state not in prev.keys():
                # print("Neighbor_state: ", neighbor_state)
                heapq.heappush(priorityq, neighbor_state)
                prev[neighbor_state] = state
    #return path
    return path
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    # return []

def astar_multi(maze):
    
    path = [maze.getStart()]
    
    #E[pointA, pointB] = actual min distance from point A to B
    E = {}
    
    #g[(cur, goals)] = lowest cost from start to cur
    g = {}
    g[(maze.getStart(), tuple(maze.getObjectives()))] = 0
    
    #h[(cur, goals)] = approx cost to end from cur
    h = {}
    h[(maze.getStart(), tuple(maze.getObjectives()))] = spanning_astar(maze, maze.getObjectives(), maze.getStart(), E)
    
    #f[(cur, goals)] = g[cur] + h[cur] --> estimate of total cost
    # f = {}
    # f[(maze.getStart(), maze.getObjectives)] =
    
    # state = (f, g, cur, remaining goals)
    start = (h[(maze.getStart(), tuple(maze.getObjectives()))], 0, maze.getStart(), tuple(maze.getObjectives()))
    
    prev = {}
    prev[start] = "start"
    
    priorityq = []
    #heapq.heapify(priorityq)
    heapq.heappush(priorityq, start)
    
    while len(priorityq) > 0:
        state = heapq.heappop(priorityq)
        cur = state[2]
        goals = state[3]
        
        if cur in goals:
            #pop cur from goals
            old_goals = goals
            new_goals = []
            for goal in goals:
                if goal != cur:
                    new_goals.append(goal)
            goals = tuple(new_goals)
            
            g[(cur, goals)] = g[(cur, old_goals)]
            h[(cur, goals)] = h[(cur, old_goals)]
            prev[(g[(cur, goals)]+h[(cur, goals)], g[(cur, goals)], cur, goals)] = prev[(g[(cur, old_goals)]+h[(cur, old_goals)], g[(cur, old_goals)], cur, old_goals)]
        
        if len(goals) == 0:
            #end reached
            path = path + ConstructPath_State(prev, state)
            break
        # print("Cur: ", cur)
        neighbors = maze.getNeighbors(cur[0], cur[1])
        for neighbor in neighbors:
            # update g[neighbor, goals]
            if (neighbor, goals) not in g.keys():
                g[(neighbor, goals)] = g[(cur, goals)] + 1
            elif (g[(cur, goals)] + 1) < g[(neighbor, goals)]:
                g.pop((neighbor, goals), None)
                g[(neighbor, goals)] = g[(cur, goals)] + 1
            # update h[neighbor, goals]
            if (neighbor, goals) not in h.keys():
                h[(neighbor, goals)] = spanning_astar(maze, goals, cur, E)
            
            neighbor_state = (h[(neighbor, goals)] + g[(neighbor, goals)], g[(neighbor, goals)], neighbor, goals)
            
            if neighbor_state not in priorityq and neighbor_state not in prev.keys():
                # print("Neighbor_state: ", neighbor_state)
                heapq.heappush(priorityq, neighbor_state)
                prev[neighbor_state] = state
    #return path
    return path
    
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # return []


def fast(maze):
    scale = 1.5
    
    path = [maze.getStart()]
    
    #E[pointA, pointB] = actual min distance from point A to B
    E = {}
    
    #g[(cur, goals)] = lowest cost from start to cur
    g = {}
    g[(maze.getStart(), tuple(maze.getObjectives()))] = 0
    
    #h[(cur, goals)] = approx cost to end from cur
    h = {}
    h[(maze.getStart(), tuple(maze.getObjectives()))] = scale * spanning_astar(maze, maze.getObjectives(), maze.getStart(), E)
    
    #f[(cur, goals)] = g[cur] + h[cur] --> estimate of total cost
    # f = {}
    # f[(maze.getStart(), maze.getObjectives)] =
    
    # state = (f, g, cur, remaining goals)
    start = (h[(maze.getStart(), tuple(maze.getObjectives()))], 0, maze.getStart(), tuple(maze.getObjectives()))
    
    prev = {}
    prev[start] = "start"
    
    priorityq = []
    #heapq.heapify(priorityq)
    heapq.heappush(priorityq, start)
    
    while len(priorityq) > 0:
        state = heapq.heappop(priorityq)
        cur = state[2]
        goals = state[3]
        
        if cur in goals:
            #pop cur from goals
            old_goals = goals
            new_goals = []
            for goal in goals:
                if goal != cur:
                    new_goals.append(goal)
            goals = tuple(new_goals)
            
            g[(cur, goals)] = g[(cur, old_goals)]
            h[(cur, goals)] = h[(cur, old_goals)]
            prev[(g[(cur, goals)]+h[(cur, goals)], g[(cur, goals)], cur, goals)] = prev[(g[(cur, old_goals)]+h[(cur, old_goals)], g[(cur, old_goals)], cur, old_goals)]
        
        if len(goals) == 0:
            #end reached
            path = path + ConstructPath_State(prev, state)
            break
        # print("Cur: ", cur)
        neighbors = maze.getNeighbors(cur[0], cur[1])
        for neighbor in neighbors:
            # update g[neighbor, goals]
            if (neighbor, goals) not in g.keys():
                g[(neighbor, goals)] = g[(cur, goals)] + 1
            elif (g[(cur, goals)] + 1) < g[(neighbor, goals)]:
                g.pop((neighbor, goals), None)
                g[(neighbor, goals)] = g[(cur, goals)] + 1
            # update h[neighbor, goals]
            if (neighbor, goals) not in h.keys():
                h[(neighbor, goals)] = scale * spanning_astar(maze, goals, cur, E)
            
            neighbor_state = (h[(neighbor, goals)] + g[(neighbor, goals)], g[(neighbor, goals)], neighbor, goals)
            
            if neighbor_state not in priorityq and neighbor_state not in prev.keys():
                # print("Neighbor_state: ", neighbor_state)
                heapq.heappush(priorityq, neighbor_state)
                prev[neighbor_state] = state
    #return path
    return path
    
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # return []
