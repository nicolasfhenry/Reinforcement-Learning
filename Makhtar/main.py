# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:11:37 2017
@author: nicolas henry
"""

#import tensorflow as tf
import numpy as np
import multiprocessing  

import matplotlib.pyplot as plt
import time
import copy

import os 
#os.chdir('C:/Users/Makhtar Ba/Documents/GitHub/Reinforcement-Learning')


from neuralnets import NeuralNetwork
from useful_func import *








def worker(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global sigma
    global env
    
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
    '''
       With gram schmidt orthogonalization
    
    '''
    '''
       But watch out the number of the workers is fixed then: Howdo we approximate the gradient then ? 
    
    epsilon_wo=gram_schmidt(epsilon_wo)
    epsilon_wi=gram_schmidt(epsilon_wi)
    
    '''
    
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    #
    initial_positions=[[ 0.99944425, -0.03333454,  0.99606243, -0.08865463, -0.0176822 ,
        0.05332311],[ 0.99990637,  0.01368383,  0.99981907,  0.01902162,  0.02523626,
       -0.02548034],[  9.99192425e-01,   4.01808159e-02,   9.98170503e-01,
         6.04619401e-02,   2.84129862e-02,   2.68182438e-04],[ 0.99940723,  0.0344266 ,  0.99632547,  0.08564789, -0.04982971,
        0.07554994],[  9.99516909e-01,  -3.10797289e-02,   9.99999914e-01,
         4.15201486e-04,  -3.08654909e-02,  -4.65136501e-02]]
    np.random.shuffle(initial_positions)
    initial_observation=initial_positions[0]
    
    #print('calling the function')
    reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    #print('Walked the steps')
    
    return(reward_worker,epsilon_wi,epsilon_wo)
    
def main(seeds,params):
    
    #input_worker = list(zip(seeds,[params]*len(seeds)))
    input_worker = list(zip(seeds,[copy.deepcopy(params) for i in range(len(seeds))]))

    pool = multiprocessing.Pool(4)
    #print('pooled')
    results = pool.map(worker,input_worker)
    
    #print('sent')

    pool.close()
    pool.join()
    
    '''
    for worker in range(len(seeds)):
        results=worker(input_worker[worker])
    '''
    return(results)

#%%
if __name__ == "__main__":
        #Params:
    
    env, num_obs, num_action = initGym()
    
    num_episodes = 20
    
    
    #Acrobot
    alpha = [0.1*1/(1+0.01*i) for i in range(num_episodes)] #parameter gradient
    sigma = 0.5 #parameter noise -update Fi
    num_workers=100
    ################
    
    
    """
    #Mountain Cart
    alpha = 10 #parameter gradient
    sigma = 0.5 #parameter noise -update Fi
    num_workers=100
    ################
    """
    
    #Initialization of the neural net for the game
    numInput=num_obs 
    numOutput=num_action
    numHidden=8 # 8 neurons per Hidden layer
    
    dim_input_hidden=numInput*numHidden
    dim_hidden_output=numHidden*numOutput
        
    #General parameters
    params = [np.random.randn(numInput,numHidden),np.random.randn(numHidden,numOutput)]
     
    reward_episode=[]

    for episode in range (num_episodes):
        print('episode : ',episode)
    
        seeds = np.random.randint(10000,size=num_workers)
        
        reward_workers,epsilon_wi,epsilon_wo =  [list(x) for x in  zip(*main(seeds,params))]
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        index_sort = np.argsort(reward_workers)
        reward_workers = np.sort(reward_workers)
        #fitness = fitness_shaping_paper(reward_workers)
        
        print(reward_workers)
        epsilon_wi = [epsilon_wi[i] for i in index_sort]
        epsilon_wo = [epsilon_wo[i] for i in index_sort]
        
        
        #grad1:
        params[0] = params[0] - alpha*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        params[1] = params[1] - alpha*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        
        '''
        #grad1:
        params[0] = params[0] + alpha[episode]*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_workers)])
        params[1] = params[1] + alpha[episode]*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_workers)])
        '''        
        
        
        print(reward_episode[-1][0])
    print(reward_episode)   
    
    #%%
    '''
    plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-v2')
    ### Test:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    NN.wo=params[1]
runNN(NN, env)
''' 