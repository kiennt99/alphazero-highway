import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from math import *
import random
import tensorflow as tf
from model import PolicyP, PolicyV

c = 1.0

class Node:    
    
    '''
    The Node class represents a node of the MCTS tree. 
    It contains the information needed for the algorithm to run its search.
    It stores extra information about neural network value and policy for that state.
    '''

    def __init__(self, game, done, parent, observation, action_index, policy_v, policy_p):
          
        # child nodes
        self.child = None
        
        # total rewards from MCTS exploration
        self.T = 0
        
        # visit count
        self.N = 0        
                
        # the environment
        self.game = game
        
        # observation of the environment
        self.observation = observation
        
        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent
        
        # action index that leads to this node
        self.action_index = action_index
        
        # the value of the node according to nn
        self.nn_v = 0
        
        # the next probabilities
        self.nn_p = None     

        self.policy_v = policy_v
        self.policy_p = policy_p
        
        
    def getUCBscore(self):        
        
        '''
        This is the formula that gives a value to the node.
        The MCTS will pick the nodes with the highest value.        
        '''
        
        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')
        
        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
                        
        value_score = (self.T / self.N) 
            
        prior_score = c * self.parent.nn_p[self.action_index] * sqrt(log(top_node.N) / self.N)
                
        # We use both the Value(s) and Policy from the neural network estimations for calculating the node value
        return value_score + prior_score
    
    
    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None
       
        
    def create_child(self, game_actions=5):
        
        '''
        We create one children for each possible action of the game, 
        then we apply such action to a copy of the current node enviroment 
        and create such child node with proper information returned from the action executed
        '''
        
        if self.done:
            return
    
        actions = []
        for i in range(game_actions):
            actions.append(i)  
        
        games = []
        for i in range(len(actions)):            
            new_game = deepcopy(self.game)
            games.append(new_game) 
 
        child = {} 
        action_index = 0
        for action, game in zip(actions, games):
            observation, reward, done, _, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action_index, self.policy_v, self.policy_p)                        
            action_index += 1
            
        self.child = child
                
            
    def explore(self):
        
        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the AlphaZero formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        '''
        
        # find a leaf node by choosing nodes with max U.
        
        current = self
        
        while current.child:

            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
            if len(actions) == 0:
                print("error zero length ", max_U)                      
            action = random.choice(actions)
            current = child[action]
            
        # play a random game, or expand if needed          
            
        if current.N < 1:
            current.nn_v, current.nn_p = current.rollout()
            current.T = current.T + current.nn_v
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.nn_v, current.nn_p = current.rollout()
            current.T = current.T + current.nn_v
            
        current.N += 1      
                
        # update statistics and backpropagate
            
        parent = current
            
        while parent.parent:
            
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T           
            
            
    def rollout(self):
        
        '''
        The rollout is where we use the neural network estimations to approximate the Value and Policy of a given node.
        With the trained neural network, it will give us a good approximation even in large state spaces searches.
        '''
        
        if self.done:
            return 0, None        
        else:
            obs = np.array([self.observation[0]]) if isinstance(self.observation[1], dict) else np.array([self.observation])
            obs = np.array([tf.reshape(obs, [-1])])
            

            v = self.policy_v(obs)
            p = self.policy_p(obs)
            
            return v.numpy().flatten()[0], p.numpy().flatten()  

    def update_tree(self, action_index):
        if not self.child:
            raise ValueError("cannot update leaf node")
        for child in self.child:
            if child.action_index == action_index:
                self = child
                self.parent = None
                return
            
    def next(self):
        
        ''' 
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            return None, None, None, None, None, True
        
        child = self.child
        
        max_N = max(node.N for node in child.values())
        
        probs = [ node.N / max_N for node in child.values() ]
        probs /= np.sum(probs)
        
        next_children = random.choices(list(child.values()), weights=probs)[0]
        
        return next_children, next_children.action_index, next_children.observation, probs, self.observation, False