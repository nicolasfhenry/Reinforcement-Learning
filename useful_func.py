# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:37:00 2017

@author: nicolas
"""

import gym
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y)), the way we use this y is already a sigmoid

def dsigmoid(y):
    return y * (1.0 - y)

def fitness_shaping(rewards):
    size = len(rewards)
    utility=[(10*c+1)/(10* size) for c in range(size)]
    
    return utility
    
def fitness_shaping_cheat(rewards):
    for i in range(len(rewards)):
        if rewards[i]==min(rewards):
            utility.append(0.01)
        else :
            utility.append((10*i+1)/(10* len(rewards)))
    
    return utility


def fitness_shaping_mountainCar(rewards):
    utility=[3*c+1 for c in range(len(rewards))]
    
    return utility
 

def initGym():
    env=gym.make('Acrobot-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    return env, obs_dim , act_dim


    
def episodeRoute(rnn, env, observation, steps=300):
    rewards = []
    inputs = env.reset()
    cum_reward = 0.0
    for j in range(steps):
        #env.render()
        #print(observation)
        outputs = rnn.feedForward(inputs)
        action = np.argmax(outputs)
        inputs, reward, done, info = env.step(action)
        if done:
            break
        cum_reward += reward
        rewards.append(cum_reward) #useless, can be interesting if we want to plot the evolution of cum reward   
    return cum_reward

def runNN(rnn, env):
    initial_observation = env.reset()
    observation=initial_observation
    for t in range(1000):
        env.render()
        #print(observation)
        
        outputs = rnn.feedForward(observation)  
        #print(rnn.feedForward(observation))
        
        action = np.argmax(outputs)
        #print(action)
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break