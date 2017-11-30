# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:59:32 2017

@author: Makhtar Ba
"""

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

num_episodes = 50


#Acrobot
alphaValue = 0.1 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100
################

def alpha(i, alphaValue):
    whenDecay=int(num_episodes/2)
    if (i<whenDecay):
        return(alphaValue)
    elif (i>=whenDecay)&(i<whenDecay*2-10):
        return(alphaValue*(1-(i-whenDecay)/whenDecay))
    else:
        return(alphaValue*(11/whenDecay))

plt.plot([alpha(i, 0.1) for i in range(200)])

"""
#Mountain Cart
alpha = 10 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100

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
    
    #Local:
    
    seed = int(input_worker[0])
    p = input_worker[1]
    
    env.seed(seed) 
    np.random.seed(seed)
    
    initial_positions=[[ 0.99944425, -0.03333454,  0.99606243, -0.08865463, -0.0176822 ,
        0.05332311],[ 0.99990637,  0.01368383,  0.99981907,  0.01902162,  0.02523626,
       -0.02548034],[  9.99192425e-01,   4.01808159e-02,   9.98170503e-01,
         6.04619401e-02,   2.84129862e-02,   2.68182438e-04],[ 0.99940723,  0.0344266 ,  0.99632547,  0.08564789, -0.04982971,
        0.07554994],[  9.99516909e-01,  -3.10797289e-02,   9.99999914e-01,
         4.15201486e-04,  -3.08654909e-02,  -4.65136501e-02]]
    np.random.shuffle(initial_positions)
    initial_observation=initial_positions[0]
    
    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    
    NN.wi=p[0]
    NN.wo=p[1]
    
    #distortions
    
    epsilon = np.random.multivariate_normal(np.zeros(dim_hidden_output+dim_input_hidden),np.identity(dim_hidden_output+dim_input_hidden))
    epsilon_wo= epsilon[0:dim_hidden_output].reshape((numHidden,numOutput))
    epsilon_wi= epsilon[dim_hidden_output:dim_hidden_output+dim_input_hidden+1].reshape((numInput,numHidden))

    
    #epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((numHidden,numOutput))
    #epsilon_wi = np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((numInput,numHidden))
    
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    #initial_observation=env.reset()
    
    reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    
    
    return(reward_worker,epsilon_wi,epsilon_wo)


def main(seeds,params):
    
    
    input_worker = list(zip(seeds,[params]*len(seeds)))
    
    pool = multiprocessing.Pool(1)
    results = pool.map(worker,input_worker)

    pool.close()
    pool.join()
    
    
    '''
       Semi random code 
       
    '''

    '''
    results=[]
    for seed in seeds:
        input_worker=list([seed,params])  
        results.append(worker(input_worker))
    '''    
    return(results)

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
        
        
        print("moy reward:")
        print(np.mean(reward_workers))
        
        epsilon_wi = [epsilon_wi[i] for i in index_sort]
        epsilon_wo = [epsilon_wo[i] for i in index_sort]
        
        
        #grad1:
        
        print("param")
        print(params[0])
        print("update:")
        print(alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)]))
        

        params[0] = params[0] - alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        params[1] = params[1] - alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        """
        #grad1:
        params[0] = params[0] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_workers)])
        params[1] = params[1] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_workers)])
        """
        
        
        #print(reward_episode[-1][0])
    print(reward_episode)   

    plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-v2')
    ### Test:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    NN.wo=params[1]
#runNN(NN, env) 