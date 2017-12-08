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

#import matplotlib.pyplot as plt
import time

from neuralnets import NeuralNetwork_2_nobias
from useful_func_mujoco import *
from scipy.linalg import toeplitz


#Params:

env, num_obs, num_action = initGym()
num_episodes = 50


#Acrobot
alphaValue = 0.009 #parameter gradient
sigma = 0.08 #parameter noise -update Fi
num_workers=300
################

def alpha(i, alphaValue):
    whenDecay=int(num_episodes/3)
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
numInput=num_obs ls
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer
dim_input_hidden=numInput*numHidden
dim_hidden_output=numHidden*numOutput
    
'''

'''
   Initialization of the two layer neural net for the game
'''

numInput=num_obs 
numOutput=num_action
numHidden1=num_obs*10 # 8 neurons per Hidden layer
numHidden2=num_obs*10
print(numHidden1,numHidden2)

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


def worker_toeplitz(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden1,numHidden2
    global dim_hidden2_output, dim_input_hidden1, dim_hidden1_hidden2
    global sigma
    global env
    
    #Local:
    
    seed = int(input_worker[0])
    seed_env = int(input_worker[1])
    
    p = input_worker[2]
    
    env.seed(seed_env) 
    
    np.random.seed(seed)
    
    #Neural Networks:
    NN = NeuralNetwork_2_nobias(numInput,numHidden1,numHidden2,numOutput)
    '''
    NN.wi=toeplitz(p[0][:numInput+1],p[0][numInput+1:])
    NN.wh=toeplitz(p[1][:numHidden1+1],p[1][numHidden1+1:])
    NN.wo=toeplitz(p[2][:numHidden2+1],p[2][numHidden2+1:])
    '''
    NN.wi=toeplitz(p[0][:numInput],p[0][numInput:])
    NN.wh=toeplitz(p[1][:numHidden1],p[1][numHidden1:])
    NN.wo=toeplitz(p[2][:numHidden2],p[2][numHidden2:])
    
    
    #distortions
    epsilon_wi = np.random.multivariate_normal([0 for x in range((numInput)+(numHidden1))],np.identity((numInput)+(numHidden1)))
    epsilon_wh = np.random.multivariate_normal([0 for x in range((numHidden1)+(numHidden2))],np.identity((numHidden1)+(numHidden2)))
    epsilon_wo = np.random.multivariate_normal([0 for x in range((numHidden2)+(numOutput))],np.identity((numHidden2)+(numOutput)))
    
    
    
    #parameters update
    NN.wi=NN.wi+sigma*toeplitz(epsilon_wi[:numInput],epsilon_wi[numInput:])#remark:we should merge the two, and reshape the matrix
    NN.wh=NN.wh+sigma*toeplitz(epsilon_wh[:numHidden1],epsilon_wh[numHidden1:])
    NN.wo=NN.wo+sigma*toeplitz(epsilon_wo[:numHidden2],epsilon_wo[numHidden2:])
    
    initial_observation=env.reset()
    
    reward_worker=episodeRoute(NN,env,initial_observation,steps=1000)
    
    
    return(reward_worker,epsilon_wi,epsilon_wh,epsilon_wo)



def main(seeds,params):
    num_workers=len(seeds)
    
    '''
    initialization=[1,2,3,4,5,6]
    
    seeds=[]
    for i in range(num_workers):
        np.random.shuffle(initialization)
        seeds.append(initialization[0])
    '''
    
    seeds_env=[0 for x in range(num_workers)]        
    input_worker = list(zip(seeds,seeds_env,[params]*len(seeds)))
    
    pool = multiprocessing.Pool(24)
    results = pool.map(worker_toeplitz,input_worker)

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
    
    
    #General parameters
    '''
       for Toeplitz Matrices 
    '''
    params = [0.01*np.random.randn(numInput+numHidden1),0.01*np.random.randn(numHidden1+numHidden2),0.01*np.random.randn(numHidden2+numOutput)] # Adding +1 is to counter the bias in the correponding layer 
    #params = [np.zeros(numInput+numHidden1),np.zeros(numHidden1+numHidden2),np.zeros(numHidden2+numOutput)] # Adding +1 is to counter the bias in the correponding layer 
     
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
        
        print("median reward:")
        print(np.median(reward_workers))
        
        print("max reward:")
        print(np.max(reward_workers))
        
        print("min reward:")
        print(np.min(reward_workers))
        
        epsilon_wi = [epsilon_wi[i] for i in index_sort]
        epsilon_wo = [epsilon_wo[i] for i in index_sort]
        epsilon_wh = [epsilon_wh[i] for i in index_sort]
        
        
        #grad1:
        
        print("param")
        print(params[0])
        print("update:")
        print(alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)]))
        print("fitness:")
        print(fitness)
        print('')
        
        '''
        params[0] = params[0] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        params[1] = params[1] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wh,reward_workers,fitness)])
        params[2] = params[2] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        '''
        
        #grad1:
        
        params[0] = params[0] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_workers)])
        params[1] = params[1] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wh,reward_workers)])
        params[2] = params[2] + alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_workers)])
        
        
        
        #print(reward_episode[-1][0])
    print(reward_episode)   


    #plt.plot([x[0] for x in reward_episode])
    save_obj(params,'params-v2Layers_{}_{}_{}_{}_{}_+'.format(num_episodes,alphaValue,sigma,np.mean(reward_workers),np.median(reward_workers)))
    ### Test:
    '''
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    NN.wo=params[1]
    '''
    '''
    initial_observation=env.reset()    
    simulation_rewards=[]
    for i in range(1000):
        simulation_rewards.append(episodeRoute(NN,env,initial_observation,steps=250))
    plt.plot(simulation_rewards)
    runNN(NN, env) 
    '''