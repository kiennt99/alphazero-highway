from copy import deepcopy
import random
from tqdm import tqdm

MCTS_POLICY_EXPLORE = 200

def Policy_Player_MCTS(mytree):  
    
    '''
    Our strategy for using AlphaZero is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to AlphaZero, is the best possible next action
    '''
    
    for i in tqdm(range(MCTS_POLICY_EXPLORE), desc="MCTS Exploration", leave=True):
        mytree.explore()
        
    next_tree, next_action, obs, p, p_obs = mytree.next()
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()
    
    return next_tree, next_action, obs, p, p_obs