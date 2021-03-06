
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:37:00 2017
@author: nicolas
"""
from __future__ import division
import gym
import numpy as np
import math

import os
import pickle


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))
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

def fitness_shaping_tanh(rewards):
    temp=[math.tanh(-(rewards[i]-np.mean(rewards))*0.01) for i in range(len(rewards))]
    return temp

 
    
def fitness_shaping_paper(rewards):
    length=len(rewards)
    temp=[max(0.0,(math.log(length/2+1)-math.log(length-(s)+1)))/(sum([max(0.0,math.log(length/2+1)-math.log(j+1)) for j in range(length)])) for s in range(length)]
    temp=[10*(x-1/length) for x in temp]
    return temp

def initGym():
    env=gym.make('Acrobot-v1')
    #env=gym.make('Asteroids-v0')
    
    #env=gym.make('CartPole-v0')
    #env=gym.make('Pendulum-v0')
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
        
        
        
def save_obj(obj, name):
    WD = os.getcwd()
    with open(WD +'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    WD = os.getcwd()
    with open(WD + '/'+ name + '.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        
        return p


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return basis