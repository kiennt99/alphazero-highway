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
import os
from tqdm import tqdm

GAME_NAME = 'highway-v0'
BUFFER_SIZE = int(1000)   # replay buffer size
BATCH_SIZE = 128          # minibatch size
UPDATE_EVERY = 1
MCTS_POLICY_EXPLORE = 200


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

def standardize_input(input): 
    input = np.array([input]) if isinstance(input[1], dict) else np.array([input])
    input = np.array([tf.reshape(input, [-1])])
    return input

# Initialize log directory for saving models
if not os.path.exists("models"):
    os.makedirs("models")

game = gym.make(GAME_NAME, config=config)
observation = game.reset()
done = False
    
new_game = deepcopy(game)
mytree = Node(new_game, False, 0, observation, 0, policy_v=policy_v, policy_p=policy_p)

for e in tqdm(range(episodes), desc="Training Episodes", leave=True):  # Progress bar for episodes

    print(f"\n### Episode {e + 1}/{episodes} ###\n")
    reward_e = 0


    obs = []
    ps = []
    p_obs = []

    step = 0

    # Simulation Process
    print(f"Starting simulation for Episode {e + 1}...")
    print(f"Step Running MCTS rollout...")
    # mytree, action, ob, p, p_ob = Policy_Player_MCTS(mytree)
    # Rollout using MCTS
    for i in tqdm(range(MCTS_POLICY_EXPLORE), desc="MCTS Exploration", leave=True):
        mytree.explore()
    while not done:
            # step += 1
            next_tree, next_action, ob, p, p_ob, is_leaf_node = mytree.next()

            if not is_leaf_node:
                obs.append(ob)
                ps.append(p)
                p_obs.append(p_ob)
                next_tree.detach_parent()
                mytree = next_tree

                # Environment interaction
                _, reward, done, _, _ = game.step(next_action)
                reward_e += reward
            # print(f"Step {step}: Reward accumulated: {reward_e}, Action taken: {action}")

            if done or is_leaf_node:
                print(f"Episode {e + 1} finished with total reward: {reward_e}")
                for i in range(len(obs)):
                    replay_buffer.add(obs[i], reward_e, p_obs[i], ps[i])
                game.close()
                break   
                 
    print(f"replay buffer length after episode {e+1}: {len(replay_buffer)}")
    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

    if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:

        print(f"\nUpdating networks after {e + 1} episodes...")
        for _ in range(10):  # Clear output for better logging
            clear_output(wait=True)

        # Training Process
        experiences = replay_buffer.sample()

        # Training Value Network
        print("Training value network (policy_v)...")
        inputs = [standardize_input(experience.obs) for experience in experiences]

        # inputs = [[experience.obs] for experience in experiences]
        targets = [[experience.v / MAX_REWARD] for experience in experiences]

        # inputs = np.array(inputs)
        inputs = np.array(inputs)
        targets = np.array(targets)

        loss_v = policy_v.train_on_batch(inputs, targets)
        v_losses.append(loss_v)
        print(f"Value network loss: {loss_v[0]:.4f}")

        # Training Policy Network
        print("Training policy network (policy_p)...")
        inputs = [standardize_input(experience.p_obs) for experience in experiences]
        targets = [[experience.p] for experience in experiences]

        inputs = np.array(inputs)
        targets = np.array(targets)

        loss_p = policy_p.train_on_batch(inputs, targets)
        p_losses.append(loss_p)
        print(f"Policy network loss: {loss_p[0]:.4f}")

        # Plot Progress
        print("Plotting metrics...")
        plt.plot(rewards)
        plt.plot(moving_average)
        plt.title("Rewards and Moving Average")
        plt.show()

        plt.plot(v_losses)
        plt.title("Value Network Losses")
        plt.show()

        plt.plot(p_losses)
        plt.title("Policy Network Losses")
        plt.show()

        print(f"Recent Moving Average Reward: {np.mean(rewards[-20:]):.2f}")

# Save the trained models at the end
print("\nTraining complete. Saving models...")
policy_v.save("models/policy_v_model.h5")
policy_p.save("models/policy_p_model.h5")
print("Models saved in the 'models' directory.")