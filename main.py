# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:11:37 2017

@author: nicolas henry
"""

import tensorflow as tf
import numpy as np
import multiprocessing  

import matplotlib.pyplot as plt
import time

from neuralnets import NeuralNetwork
from useful_func import *


#Params:

env, num_obs, num_action = initGym()
initial_observation=env.reset()
num_episodes=20

alpha = 1e-5 #parameter gradient
sigma = 1 #parameter noise -update Fi
num_workers=50

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer

dim_input_hidden=numInput*numHidden
dim_hidden_output=numHidden*numOutput
    

params = [np.random.randn(numInput,numHidden),np.random.randn(numHidden,numOutput)]



def worker(seed):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global params, sigma
    global env
    global initial_observation
    
    np.random.seed(seed)
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    
    NN.wi=params[0]
    NN.w0=params[1]
    
    #distortions
    epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((numHidden,numOutput))
    epsilon_wi=np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((numInput,numHidden))
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    #
    reward_worker=episodeRoute(NN,env,initial_observation)
    
    
    return(reward_worker,epsilon_wi,epsilon_wo)


def main(seeds):
    
    
    pool = multiprocessing.Pool(4)
    results = pool.map(worker,seeds)

    pool.close()
    pool.join()

    return(results)

if __name__ == "__main__":
    #General parameters
    
    reward_episode=[]
    seed = 0
    for i in range (num_episodes):
        print('episode : ',i)
        initial_observation=env.reset()
        reward_workers=[]
        incremental_gradient_wo=0
        incremental_gradient_wi=0
        np.random.seed(50)
        seeds = np.random.randint(10000,size=num_workers)
        reward_workers,epsilon_wi,epsilon_wo =  [list(x) for x in  zip(*main(seeds))]
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        params[0] = params[0] + alpha*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_worker)])
        params[1] = params[1] + alpha*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_worker)])
        
        seed+=1
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
    #runNN(NN[1], env)    