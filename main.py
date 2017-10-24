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
num_episodes = 100

#Acrobot
alpha = 0.01 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100
################

"""
#Mountain Cart
alpha = 0.01 #parameter gradient
sigma = 10 #parameter noise -update Fi
num_workers=100
################
"""

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer

dim_input_hidden=numInput*numHidden
dim_hidden_output=numHidden*numOutput
    





def worker(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global sigma
    global env
    global initial_observation
    #Lovcal:
    
    seed = input_worker[0]
    p = input_worker[1]
    
    np.random.seed(seed)
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    
    NN.wi=p[0]
    NN.wo=p[1]
    
    #distortions
    epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((numHidden,numOutput))
    epsilon_wi = np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((numInput,numHidden))
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    #
    reward_worker=episodeRoute(NN,env,initial_observation,steps=750)
    
    
    return(reward_worker,epsilon_wi,epsilon_wo)


def main(seeds,params):
    
    input_worker = list(zip(seeds,[params]*len(seeds)))
    
    pool = multiprocessing.Pool(4)
    results = pool.map(worker,input_worker)

    pool.close()
    pool.join()

    return(results)
#%%
if __name__ == "__main__":
    #General parameters
    params = [np.random.randn(numInput,numHidden),np.random.randn(numHidden,numOutput)]
      
    reward_episode=[]
    
    for i in range (num_episodes):
        print('episode : ',i)
        
        
        
        
        seeds = np.random.randint(10000,size=num_workers)
        
        reward_workers,epsilon_wi,epsilon_wo =  [list(x) for x in  zip(*main(seeds,params))]
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        index_sort = np.argsort(reward_workers)
        reward_workers = np.sort(reward_workers)
        fitness = fitness_shaping_paper(reward_workers)

        print(reward_workers)
        epsilon_wi = [epsilon_wi[i] for i in index_sort]
        epsilon_wo = [epsilon_wo[i] for i in index_sort]
        
        #grad1:
        #params[0] = params[0] - alpha*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        #params[1] = params[1] - alpha*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        #grad2:
        
        params[0] = params[0] - alpha*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        
        
        #print([F*w for F,w in zip(reward_workers,fitness)])
        #print(reward_workers)
        print(fitness)
        
        
        print(reward_episode[-1][0])
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-mountaincar')
    ### Test:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    NN.wo=params[1]
    runNN(NN, env)    