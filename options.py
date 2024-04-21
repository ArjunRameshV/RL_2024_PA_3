'''
This file contains some variants of options, that reporesent the temporal abstractions
'''

import numpy as np
from algorithms import epsilon_greedy_policy,  update_epsilon

class Option:
  def __init__(self, env, goal_state, epsilon=0.1, alpha=0.1, gamma=0.99):
    self.env = env
    self.goal_state = goal_state

    self.q_value = np.zeros((env.observation_space.n//20, env.action_space.n - 2))
    
    self.epsilon = epsilon
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.99
    self.alpha = alpha
    self.gamma = gamma

  def is_done(self, state):
    x, y, _, _ = self.env.decode(state)
    return x == self.goal_state[0] and y == self.goal_state[1]
  
  def _update_epsilon(self):
    self.epsilon = update_epsilon(self.epsilon, self.epsilon_min, self.epsilon_decay)

  def update_policy(self, state, action, reward, next_state):
    x, y, _, _ = self.env.decode(state)
    nx, ny, _, _ = self.env.decode(next_state)
    self.q_value[5*x+y, action] += self.alpha * (reward + self.gamma * np.max(self.q_value[5*nx+ny, :]) - self.q_value[5*x+y, action])

  def select_action(self, state):
    x, y, passegner_loc, dest_loc = self.env.decode(state)
    if self.is_done(state):
      if passegner_loc == self.env.unwrapped.locs.index(self.goal_state):
        return (4, True)
      elif dest_loc == self.env.unwrapped.locs.index(self.goal_state):
        return (5, True)
      else:   
        if self.env.unwrapped.locs.index(self.goal_state) in [0, 1]:
          return (1, True)
        return (0, True)
    else:
      option_action = epsilon_greedy_policy(self.q_value, 5*x+y, epsilon=self.epsilon)
      self._update_epsilon()
      return (option_action, False)