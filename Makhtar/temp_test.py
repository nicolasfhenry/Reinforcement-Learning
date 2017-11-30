# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:46:54 2017

@author: Makhtar Ba
"""

import gym
#from useful_func import *
import math
import mujoco_py

env = gym.make('Swimmer-v1')
rewards=[]
actions=[]
reward_episode=[]
for i_episode in range(20):
    observation = env.reset()
    rewards=[]
    for t in range(1000000):
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    reward_episode.append(sum(rewards))    

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space
print(env,'the Observation dimension is', obs_dim, 'The action dimension is ', act_dim.shape[0])
print('Low is', env.action_space.low, 'high is', env.action_space.high)
print('reward per episode is ', reward_episode)

