# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:11:37 2017

@author: nicolas henry
"""

import tensorflow as tf
import numpy as np
import multiprocessing
import copy  

import matplotlib.pyplot as plt
import time

from neuralnets.neuralnets import *
from useful_func import *


#Params:

env, num_obs, num_action = initGym()

num_episodes = 20


#Acrobot
alpha = 0.01 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=25
################


"""
#Mountain Cart
alpha = 0.01 #parameter gradient
sigma = 100 #parameter noise -update Fi
num_workers=50
################
"""

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer
hidden_dims=[numHidden,numHidden] #Two layers
dims = [numInput] + hidden_dims
    



def worker(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global hidden_dims
    global dims
    global sigma
    
    global env
    
    #Lovcal:
    
    seed = input_worker[0]
    
    
    np.random.seed(seed)
    #Neural Networks:
    #NN = NeuralNetwork(numInput,numHidden,numOutput)
    
    #dropout_config = {"enabled":True, "keep_prob": 0.2}
    #use_bn = True
    NN = MLP(input_dim=numInput, hidden_dims=hidden_dims, num_classes=numOutput, 
            weight_scale=1e-3, l2_reg=0.0, dropout_config=None,use_bn=False)
    
    NN.params = input_worker[1]
    
    #distortions
    epsilon = {}
    for i in range(len(hidden_dims)):
            # initialize weights and bias for the first (L-1) layer
            weight_name = "weight_{}".format(i)
            bias_name = "bias_{}".format(i)
            W = np.random.normal(size = (dims[i], dims[i + 1]))
            b = np.random.normal(size = (dims[i + 1],))
            NN.params[weight_name] += sigma*W
            NN.params[bias_name] += sigma*b
            epsilon[weight_name] = W
            epsilon[bias_name] = b
            # initalize batch normalization parameters

    # initialize weights and bias for the last layer
    weight_name = "weight_{}".format(len(hidden_dims))
    bias_name = "bias_{}".format(len(hidden_dims))
    W = np.random.normal(size =(dims[-1], numOutput))
    b = np.random.normal(size = (numOutput,))
    NN.params[weight_name] += sigma*W
    NN.params[bias_name] += sigma*b
    epsilon[weight_name] = W
    epsilon[bias_name] = b
    
    #
    initial_observation = env.reset()
    reward_worker=episodeRoute(NN,env,initial_observation,steps=750)
    
    
    return(reward_worker,epsilon)


def main(seeds,params):
    
    input_worker = list(zip(seeds,[copy.deepcopy(params) for i in range(len(seeds))]))
    
    pool = multiprocessing.Pool(4)
    results = pool.map(worker,input_worker)

    pool.close()
    pool.join()

    return(results)
#%%
if __name__ == "__main__":
    #General parameters
    params = MLP(input_dim=numInput, hidden_dims=hidden_dims, num_classes=numOutput, 
            weight_scale=1e-3, l2_reg=0.0, dropout_config=None,use_bn=False).params
     
    reward_episode=[]
    
    for i in range (num_episodes):
        print('episode : ',i)
        
        
        
        seeds = np.random.randint(10000,size=num_workers)
        
        reward_workers,eps =  [list(x) for x in  zip(*main(seeds,params))]
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
    
        #reward_workers = np.sort(reward_workers)
        #fitness = fitness_shaping_paper(reward_workers)
        
        
        
        for i in range(len(hidden_dims)):
            # initialize weights and bias for the first (L-1) layer
            weight_name = "weight_{}".format(i)
            bias_name = "bias_{}".format(i)
            params[weight_name] += alpha*(1/(num_workers*sigma))*sum([eps[weight_name]*F for eps,F in zip(eps,reward_workers)])
            params[bias_name] += alpha*(1/(num_workers*sigma))*sum([eps[bias_name] *F for eps,F in zip(eps,reward_workers)])
            
            # initalize batch normalization parameters

        # initialize weights and bias for the last layer
        weight_name = "weight_{}".format(len(hidden_dims))
        bias_name = "bias_{}".format(len(hidden_dims))
        params[weight_name] += alpha*(1/(num_workers*sigma))*sum([eps[weight_name]*F for eps,F in zip(eps,reward_workers)])
        params[bias_name] += alpha*(1/(num_workers*sigma))*sum([eps[bias_name]*F for eps,F in zip(eps,reward_workers)])
        
                
        print(np.sort(reward_workers))
        print(reward_episode[-1][0])
    print(reward_episode)   
    #%%
    plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-v3')
    ### Test:
    NN = MLP(input_dim=numInput, hidden_dims=hidden_dims, num_classes=numOutput, 
            weight_scale=1e-3, l2_reg=0.0, dropout_config=None,use_bn=False)
    NN.params = params
    
    runNN(NN, env)    