import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        actions.reverse()
        self.actions_priority = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
    
    def discretize_state(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        
        adjoining_wall_x, adjoining_wall_y = (0,0)
        if snake_head_x == 1 * utils.GRID_SIZE:
            adjoining_wall_x = 1
        if snake_head_x == 12 * utils.GRID_SIZE:
            adjoining_wall_x = 2
        if snake_head_y == 1 * utils.GRID_SIZE:
            adjoining_wall_y = 1
        if snake_head_y == 12 * utils.GRID_SIZE:
            adjoining_wall_y = 2
        
        food_dir_x, food_dir_y = (0,0)
        if food_x < snake_head_x:
            food_dir_x = 1 #food left of head
        if food_x > snake_head_x:
            food_dir_x = 2 #food right of head
        if food_y < snake_head_y:
            food_dir_y = 1 #food above head
        if food_y > snake_head_y:
            food_dir_y = 2 #food below head
        
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = (0,0,0,0)
        for body_x, body_y in snake_body:
            if (snake_head_x == body_x) and (snake_head_y - utils.GRID_SIZE == body_y):
                adjoining_body_top = 1
            if (snake_head_x == body_x) and (snake_head_y + utils.GRID_SIZE == body_y):
                adjoining_body_bottom = 1
            if (snake_head_x - 1*utils.GRID_SIZE == body_x) and (snake_head_y == body_y):
                adjoining_body_left = 1
            if (snake_head_x + 1*utils.GRID_SIZE == body_x) and (snake_head_y == body_y):
                adjoining_body_right = 1
        
        return [adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]
    
    def reward(self, points, dead):
        if dead:
            return -1
        if points - self.points > 0:
            return 1
        return -0.1
    
    def maxQ(self, state):
        # UP, DOWN, LEFT, RIGHT = [0,1,2,3]
        # actions_priority = [3,2,1,0]
        a_max, Qmax = (-1, None)
        for a in self.actions_priority:
            s_a = tuple(state[:] + [a])
            if Qmax == None:
                Qmax = self.Q[s_a]
                a_max = a
                continue
            if self.Q[s_a] > Qmax:
                Qmax = self.Q[s_a]
                a_max = a
        return (Qmax, a_max)
    
    def updateQtable(self, cur_state, r):
        prev_state = self.discretize_state(self.s)
        prev_s_prev_a = tuple(prev_state[:] + [self.a])
        alpha = self.C/(self.C + self.N[prev_s_prev_a])
        Qmax, a = self.maxQ(cur_state)
        self.Q[prev_s_prev_a] += alpha * (r + self.gamma * Qmax - self.Q[prev_s_prev_a])
            
    def f(self, u, n):
        if n < self.Ne:
            return 1
        return u
    
    def explore(self, state):
        # UP, DOWN, LEFT, RIGHT = [0,1,2,3]
        # actions_priority = [3,2,1,0]
        amax, fmax = (-1, None)
        for a in self.actions_priority:
            s_a = tuple(state[:] + [a])
            f = self.f(self.Q[s_a], self.N[s_a])
            if fmax == None:
                fmax = f
                amax = a
                continue
            if f > fmax:
                fmax = f
                amax = a
        return amax
            
    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        
        cur_s = self.discretize_state(state)
        # Test Phase
        if not self._train: # Compute best action (which maximize Q)
            Qmax, action = self.maxQ(cur_s)
        # Training Phase
        else:
            if self.s != None: # Update q table with best action (which maximize Q)
                self.updateQtable(cur_s, self.reward(points, dead))
            action = self.explore(cur_s) # Compute next action with exploration (prepare to return this one)
            if not dead: # Update N table 
                s_a = tuple(cur_s[:] + [action])
                self.N[s_a] += 1
            self.s = state[:] # Cache of system state
            self.a = action
            self.points = points
        if dead:
            self.reset()
        
        return action
