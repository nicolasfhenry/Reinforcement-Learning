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

#import tensorflow as tf
import numpy as np
import multiprocessing  

import matplotlib.pyplot as plt
import time

from neuralnets import NeuralNetwork_2
from useful_func_mujoco import *


#Params:

env, num_obs, num_action = initGym()
num_episodes = 100


#Acrobot
alphaValue = 0.1 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100
################

def alpha(i, alphaValue):
    whenDecay=int(num_episodes/2)
    if (i<whenDecay):
        return(alphaValue)
    elif (i>=whenDecay)&(i<whenDecay*2-5):
        return(alphaValue*(1-(i-whenDecay)/whenDecay))
    else:
        return(alphaValue*(6/whenDecay))


"""
#Mountain Cart
alpha = 10 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100

"""

'''
   Initialization of the one layer neural net for the game
'''
'''

numInput=num_obs 
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer

dim_input_hidden=numInput*numHidden
dim_hidden_output=numHidden*numOutput
    
'''

'''
   Initialization of the two layer neural net for the game
'''

numInput=num_obs 
numOutput=5**num_action
numHidden1=15 # 8 neurons per Hidden layer
numHidden2=15

dim_input_hidden1=numInput*numHidden1
dim_hidden1_hidden2=numHidden1*numHidden2

dim_hidden2_output=numHidden2*numOutput
    




def worker(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_input_hidden, dim_hidden_output
    global sigma
    global env
    
    #Local:
    
    seed = int(input_worker[0])
    p = input_worker[1]
    
    env.seed(seed) 
    
    
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
    
    #initial_observation=env.reset()
    
    reward_worker=episodeRoute(NN,env,initial_observation,steps=1000)
    
    
    return(reward_worker,epsilon_wi,epsilon_wo)




def worker_2(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden1,numHidden2
    global dim_hidden2_output, dim_input_hidden1, dim_hidden1_hidden2
    global sigma
    global env
    
    #Local:
    
    seed = int(input_worker[0])
    p = input_worker[1]
    
    env.seed(seed) 
    np.random.seed(seed)
    
    #Neural Networks:
    NN = NeuralNetwork_2(numInput,numHidden1,numHidden2,numOutput)
    
    NN.wi=p[0]
    NN.wh=p[1]
    NN.wo=p[2]
    
    #distortions
    
    epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden2_output)],np.identity(dim_hidden2_output)).reshape((numHidden2,numOutput))
    epsilon_wi = np.random.multivariate_normal([0 for x in range(dim_input_hidden1)],np.identity(dim_input_hidden1)).reshape((numInput,numHidden1))
    epsilon_wh = np.random.multivariate_normal([0 for x in range(dim_hidden1_hidden2)],np.identity(dim_hidden1_hidden2)).reshape((numHidden1,numHidden2))
    
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    NN.wh=NN.wh+epsilon_wh*sigma
    
    initial_observation=env.reset()
    
    reward_worker=episodeRouteSwimmer(NN,env,initial_observation,steps=1000)
    
    
    return(reward_worker,epsilon_wi,epsilon_wh,epsilon_wo)


def main(seeds,params):
    
    initialization=[1,2,3,4,5,6]
    num_workers=len(seeds)
    seeds=[]
    for i in range(num_workers):
        np.random.shuffle(initialization)
        seeds.append(initialization[0])
        
    input_worker = list(zip(seeds,[params]*len(seeds)))
    
    pool = multiprocessing.Pool(4)
    results = pool.map(worker_2,input_worker)

    pool.close()
    pool.join()
    
    '''
       Semi random code 
       
    '''
    
    '''
    
    initialization=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    '''
    '''
    results=[]
    for seed in seeds:
        input_worker=list([seed,params])  
        results.append(worker_2(input_worker))
    '''
    
    return(results)

if __name__ == "__main__":
    
    '''
       One layer version 
    
    '''
    '''
    
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
    
    '''
    
    #General parameters
    params = [np.random.randn(numInput,numHidden1),np.random.randn(numHidden1,numHidden2),np.random.randn(numHidden2,numOutput)]
     
    reward_episode=[]
    
    for episode in range (num_episodes):
        print('episode : ',episode)
        
        
        
        seeds = np.random.randint(10000,size=num_workers)
        reward_workers,epsilon_wi,epsilon_wh,epsilon_wo =  [list(x) for x in  zip(*main(seeds,params))]
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])

        index_sort = np.argsort(reward_workers)
        reward_workers = np.sort(reward_workers)
        fitness = fitness_shaping_paper(reward_workers)
        
        
        print("moy reward:")
        print(np.mean(reward_workers))
        
        epsilon_wi = [epsilon_wi[i] for i in index_sort]
        epsilon_wo = [epsilon_wo[i] for i in index_sort]
        epsilon_wh = [epsilon_wh[i] for i in index_sort]
        
        
        #grad1:
        
        print("param")
        print(params[0])
        print("update:")
        print(alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)]))
        
        params[0] = params[0] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        params[1] = params[1] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wh,reward_workers,fitness)])
        params[2] = params[2] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        
        """
        #grad1:
        params[0] = params[0] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_workers)])
        params[1] = params[1] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_workers)])
        """
        
        
        #print(reward_episode[-1][0])
    print(reward_episode)   


    #plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-v2Layers')
    
    params=load_obj('params-v2Layers')
    
    
    ### Test:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    
    NN.wo=params[1]
    '''
    initial_observation=env.reset()    
    simulation_rewards=[]
    for i in range(1000):
        simulation_rewards.append(episodeRoute(NN,env,initial_observation,steps=250))
    plt.plot(simulation_rewards)
    runNN(NN, env) 
    '''