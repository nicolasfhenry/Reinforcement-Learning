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

from neuralnets import NeuralNetwork
from useful_func import *


#Params:

env, num_obs, num_action = initGym()

num_episodes = 50


#Acrobot
alphaValue = 0.3 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100
################

def alpha(i, alphaValue):
    whenDecay=int(num_episodes/2)
    if (i<whenDecay):
        return(alphaValue)
    elif (i>=whenDecay)&(i<whenDecay*2-40):
        return(alphaValue*(1-(i-whenDecay)/whenDecay))
    else:
        return(alphaValue*(41/whenDecay))


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
    


def pairwise_selection(reward_workers,epsilon_wi,epsilon_wo):
    new_reward_workers, new_epsilon_wi, new_epsilon_wo =[],[],[]
    mid_len=int(len(reward_workers)/2)
    for i in range(mid_len):
        if reward_workers[i]>=reward_workers[i+mid_len]:
            new_reward_workers.append(reward_workers[i])
            new_epsilon_wi.append(epsilon_wi[i])
            new_epsilon_wo.append(epsilon_wo[i])
        else:
            new_reward_workers.append(reward_workers[i+mid_len])
            new_epsilon_wi.append(epsilon_wi[i+mid_len])
            new_epsilon_wo.append(epsilon_wo[i+mid_len])
    return(new_reward_workers, new_epsilon_wi, new_epsilon_wo)
            
        


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


def worker_GS(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global sigma
    global env
    
    
    #Local:

    seed=int(input_worker[0])
    epsilon = input_worker[1]
    p = input_worker[2]
    
    env.seed(seed) 
    #np.random.seed(seed)    
    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    
    NN.wi=p[0]
    NN.wo=p[1]
    
    #distortions
    
    epsilon_wo= epsilon[0:dim_hidden_output].reshape((numHidden,numOutput))
    epsilon_wi= epsilon[dim_hidden_output:dim_hidden_output+dim_input_hidden+1].reshape((numInput,numHidden))

    
    #epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((numHidden,numOutput))
    #epsilon_wi = np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((numInput,numHidden))
    
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    initial_observation=env.reset()
    
    reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    
    return(reward_worker,epsilon_wi,epsilon_wo)


def main(seeds,GS_epsilons,params):
    
    
    input_worker = list(zip(seeds,GS_epsilons,[params]*len(GS_epsilons)))

    pool = multiprocessing.Pool(4)
    results = pool.map(worker_GS,input_worker)

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
    
    for episode in range (num_episodes):
        print('episode : ',episode)
        
        
        
        
        num_samples=dim_hidden_output+dim_input_hidden 
        num_workers=2*num_samples
        epsilons_ini = [np.random.multivariate_normal(np.zeros(dim_hidden_output+dim_input_hidden),np.identity(dim_hidden_output+dim_input_hidden)) for i in range(num_samples)]      
        GS_epsilons_ini=gram_schmidt(epsilons_ini)
        GS_epsilons_neg=[-elem for elem in GS_epsilons_ini]
        epsilons=GS_epsilons_ini+GS_epsilons_neg

        seeds = np.random.randint(10000,size=num_workers)

        reward_workers_ini,epsilon_wi_ini,epsilon_wo_ini =  [list(x) for x in  zip(*main(seeds,epsilons,params))]
        
        reward_workers,epsilon_wi,epsilon_wo = pairwise_selection(reward_workers_ini,epsilon_wi_ini,epsilon_wo_ini)
        
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
        print(alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)]))
        
        params[0] = params[0] - alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        params[1] = params[1] - alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        

        #grad1:
        '''
        params[0] = params[0] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_workers)])
        params[1] = params[1] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_workers)])
        '''
        
        #print(reward_episode[-1][0])
    print(reward_episode)   

    #plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-vGS-1')
    ### Test:
    '''
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    NN.wo=params[1]
    '''
#runNN(NN, env) 