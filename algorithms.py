'''
This file contains a series of algorithms, both utility and learning
 used to better understand options and Q-learning
'''

import gymnasium as gym
import matplotlib.pyplot as plt
import random
import numpy as np

from tqdm import tqdm
import random
import seaborn as sns
from IPython.display import HTML


####################### Utility Functions #######################

############ Used for learning ############
def epsilon_greedy_policy(q_values, state, epsilon):
    if np.random.rand() < epsilon:  # With probability epsilon, choose a random action
        action = np.random.randint(len(q_values[state]))
    else:  # With probability (1 - epsilon), choose the action with the highest Q-value for the current state
        action = np.argmax(q_values[state])
    return action 


def update_epsilon(epsilon, epsilon_min, epsilon_decay):
    return max(epsilon_min , epsilon_decay * epsilon)


def get_mod_state(env, state):
    _, _, passenger_loc, drop_loc = env.decode(state)
    return 4 * passenger_loc + drop_loc


############ Used for plotting ############
def plot_average_reward(rewards_history, iterations = 100, title = "average rewards plot", baseline = 0):
    plt.figure(figsize = (10,5))
    avg_rewards = [np.average(rewards_history[i : i + iterations]) for i in range(len(rewards_history) - iterations)]
    plt.plot(avg_rewards,label = 'Model Score')
    plt.plot([baseline for i in range(len(avg_rewards))],label = 'Baseline')
    plt.xlabel('Episodes')
    plt.ylabel('score averaged over previous 100 runs')
    plt.title(title)
    plt.legend()


def compare_learning_rewards(rewards_history_list, iterations = 100, title = "Compare rewards", baseline = 0):
    plt.figure(figsize = (20,10))

    for rewards_history, label in rewards_history_list:
        avg_rewards = [np.average(rewards_history[i : i + iterations]) for i in range(len(rewards_history) - iterations)]
        plt.plot(avg_rewards, label = label)

    plt.plot([baseline for i in range(len(rewards_history_list[0][0]))], '--', label = f'Baseline score ({baseline})')
    plt.xlabel('Episodes')
    plt.ylabel('score averaged over previous 100 runs')
    plt.title(title)
    plt.legend()


#################################################################

####################### Learning Algorithms (for Option) #######################

class OptionsQLearning:
    def __init__(self, env, options, episodes = 1500, gamma = 0.99) -> None:
        self.env = env
        self.options = options
        self.episodes = episodes
        self.gamma = gamma
        self.rewards_history = []
    
    def train(self):
        for _ in tqdm(range(self.episodes)):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0

            # While episode is not over
            while not done:
                _, _, passenger_loc, drop_loc = self.env.decode(state)

                # Go to the passenger and pick up
                passenger_found = False
                while not passenger_found and not done and passenger_loc < 4:
                    # Select action using the option
                    action, passenger_found = self.options.get(passenger_loc).select_action(state)

                    next_state, reward, done, _, _ = self.env.step(action)

                    # Calculate the accumulated reward for the option
                    episode_reward += reward

                    # update option policy
                    if action < 4:
                        self.options.get(passenger_loc).update_policy(state, action, reward, next_state)
                    state = next_state

                trip_complete = False
                while not trip_complete and not done:
                    # Select action using the option
                    action, trip_complete = self.options.get(drop_loc).select_action(state)

                    next_state, reward, done, _, _ = self.env.step(action)

                    # Calculate the accumulated reward for the option
                    episode_reward += reward

                    # update option policy
                    if action < 4:
                        self.options.get(drop_loc).update_policy(state, action, reward, next_state)

                    state = next_state

            self.rewards_history.append(episode_reward)


class SMDPQLearning:
    def __init__(
        self,
        env, 
        options, 
        q_values_state_count, 
        q_values_action_count, 
        episdoes = 1500, 
        gamma = 0.99, 
        alpha = 0.1, 
        epsilon = 0.1
    ) -> None:
        
        self.env = env
        self.options = options
        self.episodes = episdoes
        self.gamma = gamma
        self.alpha = alpha
        self.rewards_history = []

        self.q_vlaues = np.zeros((q_values_state_count, q_values_action_count))
        self.update_freq = np.zeros((q_values_state_count, q_values_action_count))

        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    
    def train(self):
        for _ in tqdm(range(self.episodes)):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0

            # While episode is not over
            while not done:
                encoded_state = get_mod_state(self.env, state)
                option_index = epsilon_greedy_policy(self.q_vlaues, encoded_state, self.epsilon)
                option = self.options.get(option_index)
                update_epsilon(self.epsilon, self.epsilon_min, self.epsilon_decay)

                reward_bar = 0
                option_done = False
                prev_state = state
                option_moves = 0
                while not option_done and not done:
                    option_action, option_done = option.select_action(state)

                    next_state, reward, done, _, _ = self.env.step(option_action)

                    episode_reward += reward
                    reward_bar = self.gamma * reward_bar + reward
                    option_moves += 1

                    if option_action < 4:
                        option.update_policy(state, option_action, reward, next_state)
                
                    state = next_state
        
                encoded_state = get_mod_state(self.env, state)
                encoded_prev_state = get_mod_state(self.env, prev_state)

                self.q_vlaues[encoded_prev_state, option_index] += self.alpha * (reward_bar + (self.gamma ** option_moves) * np.max(self.q_vlaues[encoded_state, :]) - self.q_vlaues[encoded_prev_state, option_index])
                self.update_freq[encoded_state, option_index] += 1

            self.rewards_history.append(episode_reward)


class IOQLearning:
    def __init__(
        self,
        env, 
        options, 
        q_values_state_count, 
        q_values_action_count, 
        episdoes = 1500, 
        gamma = 0.99, 
        alpha = 0.1, 
        epsilon = 0.1
    ) -> None:
        
        self.env = env
        self.options = options
        self.episodes = episdoes
        self.gamma = gamma
        self.alpha = alpha
        self.rewards_history = []

        self.q_vlaues = np.zeros((q_values_state_count, q_values_action_count))
        self.update_freq = np.zeros((q_values_state_count, q_values_action_count))

        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

    
    def train(self):
        for _ in tqdm(range(self.episodes)):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0

            # While episode is not over
            while not done:
                encoded_state = get_mod_state(self.env, state)
                option_index = epsilon_greedy_policy(self.q_vlaues, encoded_state, self.epsilon)
                option = self.options.get(option_index)
                update_epsilon(self.epsilon, self.epsilon_min, self.epsilon_decay)

                reward_bar = 0
                option_done = False
                while not option_done:
                    option_action, option_done = option.select_action(state)

                    next_state, reward, done, _, _ = self.env.step(option_action)

                    episode_reward += reward
                    reward_bar = self.gamma * reward_bar + reward

                    if option_action < 4:
                        option.update_policy(state, option_action, reward, next_state)

                    for opt_index, io_option in self.options.items():
                        io_option_action, io_option_done = io_option.select_action(state)
                        if io_option_action == option_action:
                            encoded_state = get_mod_state(self.env, state)
                            encoded_next_state = get_mod_state(self.env, next_state)
                            if io_option_done:
                                self.q_vlaues[encoded_state, opt_index] += self.alpha*(reward + self.gamma * np.max(self.q_vlaues[encoded_next_state, :]) - self.q_vlaues[encoded_state, opt_index])
                            else:
                                self.q_vlaues[encoded_state, opt_index] += self.alpha*(reward + self.gamma * self.q_vlaues[encoded_next_state, opt_index] - self.q_vlaues[encoded_state, opt_index]) 

                            self.update_freq[encoded_state, opt_index] += 1

                    state = next_state

            self.rewards_history.append(episode_reward)