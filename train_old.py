from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output
from replay_buffer import ReplayBuffer
from model import PolicyV, PolicyP
import numpy as np
import tensorflow as tf
from tensorflow import keras
from policy import Policy_Player_MCTS
from mcts import Node
from copy import deepcopy
import highway_env


GAME_NAME = 'highway-v0'
BUFFER_SIZE = int(1000)   # replay buffer size
BATCH_SIZE = 128          # minibatch size
UPDATE_EVERY = 1

episodes = 250

rewards = []
moving_average = []
v_losses = []
p_losses = []

# the maximum reward of the current game to scale the values
MAX_REWARD = 500

# Create the replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

# Create the Value neural network. 
#The loss is the Mean Squared Error between predicted and actual values.
policy_v = PolicyV()
policy_v.compile(optimizer=keras.optimizers.Adam(), 
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanSquaredError()])


# Create the Policy neural network. 
#The loss is the Categorical Crossentropy between the predicted and actual policy according to visit counts.
policy_p = PolicyP()
policy_p.compile(optimizer=keras.optimizers.Adam(), 
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=[tf.keras.metrics.CategoricalCrossentropy()])

'''
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our AlphaZero algorithm
- we will collect and plot the rewards to check if the AlphaZero is actually working.
- For CartPole-v1, in particular, 500 is the maximum possible reward. 
'''

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
}

for e in range(episodes):

    reward_e = 0    
    game = gym.make(GAME_NAME, config=config)
    observation = game.reset() 
    done = False
    
    new_game = deepcopy(game)
    mytree = Node(new_game, False, 0, observation, 0, policy_v=policy_v, policy_p=policy_p)
    
    print('episode #' + str(e+1))
    
    obs = []
    ps = []
    p_obs = []
    
    step = 0
    
    while not done:
        
        step = step + 1
    
        mytree, action, ob, p, p_ob = Policy_Player_MCTS(mytree)
        
        obs.append(ob)
        ps.append(p)
        p_obs.append(p_ob)      
            
        _, reward, done, _, _ = game.step(action)  
            
        reward_e = reward_e + reward
        
        #game.render()
        
        print('reward ' + str(reward_e))
                
        if done:
            for i in range(len(obs)):
                replay_buffer.add(obs[i], reward_e, p_obs[i], ps[i])
            game.close()
            break
        
    print('reward ' + str(reward_e))
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))
    
    if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:   
        
        # clear output
        
        for i in range(10):
            clear_output(wait=True) 
        
        # update and train the neural networks
                
        experiences = replay_buffer.sample()
            
        # Each state has as target value the total rewards of the episode
            
        inputs = [[experience.obs] for experience in experiences]
        targets = [[experience.v / MAX_REWARD] for experience in experiences]
            
        inputs = np.array(inputs)
        targets = np.array(targets)
                
        loss_v = policy_v.train_on_batch(inputs, targets)  
                        
        v_losses.append(loss_v)
        
        # Each state has as target policy the policy according to visit counts
            
        inputs = [[experience.p_obs] for experience in experiences]
        targets = [[experience.p] for experience in experiences]
            
        inputs = np.array(inputs)
        targets = np.array(targets)
                
        loss_p = policy_p.train_on_batch(inputs, targets) 
                        
        p_losses.append(loss_p)
        
        # plot rewards, value losses and policy losses
    
        plt.plot(rewards)
        plt.plot(moving_average)
        plt.show()

        plt.plot(v_losses)
        plt.show()

        plt.plot(p_losses)
        plt.show()
        
        print('moving average: ' + str(np.mean(rewards[-20:])))