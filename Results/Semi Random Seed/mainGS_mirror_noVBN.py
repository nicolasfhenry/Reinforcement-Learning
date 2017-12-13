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
from operator import add
#import matplotlib.pyplot as plt
import time
import os

from neuralnets_VBN import NeuralNetwork
from useful_func import *


#Params:

env, num_obs, num_action = initGym()

num_episodes = 50


#Acrobot
alphaValue = 0.5 #parameter gradient
sigma = 0.8 #parameter noise -update Fi
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

#plt.plot([alpha(i, 0.3) for i in range(200)])

"""
#Mountain Cart
alpha = 10 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100

"""

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action

numHidden1=7 # 14 neurons per Hidden layer
numHidden2=7

dim_input_hidden1=numInput*numHidden1
dim_hidden1_hidden2=numHidden1*numHidden2
dim_hidden2_output=numHidden2*numOutput
    


def pairwise_selection(reward_workers,epsilon_W1,epsilon_W2, epsilon_W3):
    new_reward_workers, new_epsilon_W1, new_epsilon_W2, new_epsilon_W3 =[],[],[],[]
    mid_len=int(len(reward_workers)/2)
    for i in range(mid_len):
        if reward_workers[i]>=reward_workers[i+mid_len]:
            new_reward_workers.append(reward_workers[i])
            new_epsilon_W1.append(epsilon_W1[i])
            new_epsilon_W2.append(epsilon_W2[i])
            new_epsilon_W3.append(epsilon_W3[i])
        else:
            new_reward_workers.append(reward_workers[i+mid_len])
            new_epsilon_W1.append(epsilon_W1[i+mid_len])
            new_epsilon_W2.append(epsilon_W2[i+mid_len])
            new_epsilon_W3.append(epsilon_W3[i+mid_len])
    return(new_reward_workers, new_epsilon_W1, new_epsilon_W2, new_epsilon_W3)
            
        


def worker(input_worker):
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
    NN = NeuralNetwork(numInput,numHidden1,numHidden2,numOutput, VBN_dict)
    
    NN.W1=p[0]
    NN.W2=p[1]
    NN.W3=p[2]
    
    #distortions    
   
    
    epsilon = np.random.multivariate_normal(np.zeros(dim_hidden2_output+dim_input_hidden1+dim_hidden1_hidden2),np.identity(dim_hidden2_output+dim_input_hidden1+dim_hidden1_hidden2))
    epsilon_W3= epsilon[0:dim_hidden2_output].reshape((numOutput,numHidden2))
    epsilon_W2= epsilon[dim_hidden2_output:dim_hidden2_output+dim_hidden1_hidden2].reshape((numHidden2,numHidden1))    
    epsilon_W1= epsilon[dim_hidden2_output+dim_hidden1_hidden2:dim_hidden2_output+dim_hidden1_hidden2+dim_input_hidden1].reshape((numHidden1,numInput))

    
    #parameters update
    NN.W1=NN.W1+epsilon_W1*sigma 
    NN.W2=NN.W2+epsilon_W2*sigma    
    NN.W3=NN.W3+epsilon_W3*sigma
    
    #initial_observation=env.reset()
    
    reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    
    
    return(reward_worker,epsilon_W1,epsilon_W2,epsilon_W3)


def worker_train_VBN(input_worker_VBN):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden1,numHidden2
    global dim_hidden2_output, dim_input_hidden1, dim_hidden1_hidden2
    global env
    
    #Local:

    seed=int(input_worker_VBN[0])
    p = input_worker_VBN[1]
    
    env.seed(seed) 
    #np.random.seed(seed)    
    VBN_dict = {}
    #VBN_dict['mu_i']=np.zeros((numInput,1))
    #VBN_dict['var_i']=np.ones((numInput,1))
    VBN_dict['mu_h1']=np.zeros((numHidden1,1))
    VBN_dict['var_h1']=np.ones((numHidden1,1))
    VBN_dict['mu_h2']=np.zeros((numHidden2,1))
    VBN_dict['var_h2']=np.zeros((numHidden2,1))   
    

    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden1,numHidden2,numOutput, VBN_dict)
    

    NN.W1=p[0]
    NN.W2=p[1]
    NN.W3=p[2]    
    
    #SHOULD IT BE PLACED IN THE LOOP ? CANT THINK RIGHT NOW
    sum_zh1=[0.] * numHidden1  
    sum_zh2=[0.] *numHidden2
    #sum_zi=[0.] * numInput 
    
    sum_zh1_sq=[0.] * numHidden1  
    sum_zh2_sq=[0.] *numHidden2
    #sum_zi_sq=[0.] * numInput  
    
    steps=250
    Ai = env.reset()  
    num_step=steps
    NN.use_VBN=False #we don't want to use feedforward options with VBN to compute the statistics
    
    for j in range(steps):
        
        Ao = NN.feedForward(Ai)  
                      
        sum_zh1=[sum(x) for x in zip(sum_zh1, NN.Z1)]
        sum_zh2=[sum(x) for x in zip(sum_zh2, NN.Z2)]
        #sum_zi=[sum(x) for x in zip(sum_zi, NN.Ai)]

        sum_zh1_sq=[sum(x) for x in zip(sum_zh1_sq, square(NN.Z1))]
        sum_zh2_sq=[sum(x) for x in zip(sum_zh2_sq, square(NN.Z2))]
        #sum_zi_sq=[sum(x) for x in zip(sum_zi_sq, square(NN.Ai))]
                
        action=np.argmax(Ao)      
        Ai, reward, done, info = env.step(action)
        
        if done:
            break
            num_step=j             
        
    #return(sum_zi,sum_zh1,sum_zh2,sum_zi_sq,sum_zh1_sq,sum_zh2_sq,num_step)
    return(sum_zh1,sum_zh2,sum_zh1_sq,sum_zh2_sq,num_step)


def worker_GS(input_worker):
    """Explanations"""
    #Global variables:
    
    global numInput,numOutput,numHidden1,numHidden2
    global dim_hidden2_output, dim_input_hidden1, dim_hidden1_hidden2
    global env
    global sigma
    global VBN_dict
    
    #Local:

    seed=int(input_worker[0])
    epsilon = input_worker[1]
    p = input_worker[2]
    
    env.seed(seed) 
    #np.random.seed(seed)    
    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden1,numHidden2,numOutput, VBN_dict)
    
    NN.use_VBN=False    
    
    NN.W1=p[0]
    NN.W2=p[1]
    NN.W3=p[2]    
    
    
    #distortions
    
    epsilon_W3= epsilon[0:dim_hidden2_output].reshape((numOutput,numHidden2))
    epsilon_W2= epsilon[dim_hidden2_output:dim_hidden2_output+dim_hidden1_hidden2].reshape((numHidden2,numHidden1))    
    epsilon_W1= epsilon[dim_hidden2_output+dim_hidden1_hidden2:dim_hidden2_output+dim_hidden1_hidden2+dim_input_hidden1].reshape((numHidden1,numInput))

    
    #parameters update
    NN.W1=NN.W1+epsilon_W1*sigma 
    NN.W2=NN.W2+epsilon_W2*sigma    
    NN.W3=NN.W3+epsilon_W3*sigma
    
    initial_observation=env.reset()

    reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    
    return(reward_worker,epsilon_W1,epsilon_W2,epsilon_W3)


def main(seeds,GS_epsilons,params): 
    
    input_worker = list(zip(seeds,GS_epsilons,[params]*len(GS_epsilons)))

    pool = multiprocessing.Pool(4)
    results = pool.map(worker_GS,input_worker)

    pool.close()
    pool.join()
    
    return(results)

    
def computeVBN(seeds_VBN, params):
        
    input_worker_VBN = list(zip(seeds_VBN,[params]*len(seeds_VBN)))

    pool = multiprocessing.Pool(4)
    results = pool.map(worker_train_VBN,input_worker_VBN)

    pool.close()
    pool.join()
    
    return(results)

if __name__ == "__main__":
    #General parameters
    params = [0.01 * np.random.randn(numHidden1,numInput), 0.01 * np.random.randn(numHidden2,numHidden1),0.01 * np.random.randn(numOutput,numHidden2)]  
    
    num_VBN_workers=20
    seeds_VBN = np.random.randint(10000,size=num_VBN_workers)
    
    reward_episode=[]
    alpha_update=[]     
    for episode in range (num_episodes):
        
        print('episode : ',episode)
        
        print('Virtual Batch Normalization')
        ### if too slow, change everything into np.arrays
        
        
        #sum_zi,sum_zh1,sum_zh2,sum_zi_sq,sum_zh1_sq,sum_zh2_sq,num_step_list =  [list(x) for x in  zip(*computeVBN(seeds_VBN,params))]
        sum_zh1,sum_zh2,sum_zh1_sq,sum_zh2_sq,num_step_list =  [list(x) for x in  zip(*computeVBN(seeds_VBN,params))]
               
        num_step=float(sum(num_step_list))
        normalizing_factor=1/num_step
       
        #mu_zi=[normalizing_factor*sum(i) for i in zip(*sum_zi)]
        #mu2_zi=[normalizing_factor*sum(i) for i in zip(*sum_zi_sq)]
        #var_zi=[(i-j) for i,j in zip(mu2_zi, square(mu_zi))]
        
        mu_zh1=[normalizing_factor*sum(i) for i in zip(*sum_zh1)]
        mu2_zh1=[normalizing_factor*sum(i) for i in zip(*sum_zh1_sq)]
        var_zh1=[i-j for i,j in zip(mu2_zh1, square(mu_zh1))]
        
        mu_zh2=[normalizing_factor*sum(i) for i in zip(*sum_zh2)]
        mu2_zh2=[normalizing_factor*sum(i) for i in zip(*sum_zh2_sq)]
        var_zh2=[i-j for i,j in zip(mu2_zh2, square(mu_zh2))]
        
        #save VBN statistics:
        VBN_dict = {}
        #VBN_dict['mu_i']=np.array(mu_zi)
        #VBN_dict['var_i']=np.array(var_zi)
        VBN_dict['mu_h1']=np.array(mu_zh1)
        VBN_dict['var_h1']=np.array(var_zh1)
        VBN_dict['mu_h2']=np.array(mu_zh2)
        VBN_dict['var_h2']=np.array(var_zh2)
        # here we need to find a way to pass this argument in the workers, make it a global var ?  YES        
                

        
        print('Episode Computations')        
        
        #Creating Epsilon       
        
        num_samples=dim_hidden2_output + dim_input_hidden1 + dim_hidden1_hidden2
        num_workers=2*num_samples
        epsilons_ini = [np.random.multivariate_normal(np.zeros(num_samples),np.identity(num_samples)) for i in range(num_samples)]      
        GS_epsilons_ini=gram_schmidt(epsilons_ini)
        GS_epsilons_neg=[-elem for elem in GS_epsilons_ini]
        epsilons=GS_epsilons_ini+GS_epsilons_neg
        
        #seeds = np.random.randint(10000,size=num_workers)
        #seeds = np.zeros(num_workers)
        seeds = np.random.randint(5,size=num_workers)
        
        reward_workers_ini, epsilon_W1_ini, epsilon_W2_ini, epsilon_W3_ini =  [list(x) for x in  zip(*main(seeds,epsilons,params))]
        
        reward_workers,epsilon_W1,epsilon_W2,epsilon_W3 = pairwise_selection(reward_workers_ini,epsilon_W1_ini,epsilon_W2_ini,epsilon_W3_ini)
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers),reward_workers])
        index_sort = np.argsort(reward_workers)
        reward_workers = np.sort(reward_workers)
        fitness = fitness_shaping_paper(reward_workers)
        
        
        print("moy reward:")
        print(np.mean(reward_workers))
        print("MEDIAN reward:")
        print(np.median(reward_workers))
        print("Max reward:")
        print(np.max(reward_workers))
        print("Min reward:")
        print(np.min(reward_workers))
        
        
        epsilon_W1 = [epsilon_W1[i] for i in index_sort]
        epsilon_W2 = [epsilon_W2[i] for i in index_sort]
        epsilon_W3 = [epsilon_W3[i] for i in index_sort]
        
        #grad1:
        
        print("param")
        print(params[0])
        print("update:")
        tmp = alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_W1,reward_workers,fitness)])
        print(tmp )
        
        alpha_update.append(tmp)
        
        
        params[0] = params[0] - alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_W1,reward_workers,fitness)])
        params[1] = params[1] - alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_W2,reward_workers,fitness)])
        params[2] = params[2] - alpha(episode,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_W3,reward_workers,fitness)])
        
        #grad1:
    result = {}
    result['num_worker'] = num_workers
    result['num_episodes'] = num_episodes
    result['alphaValue'] = alphaValue
    result['sigma'] = sigma
    result['reward'] = reward_episode
    result['numInput'] = numInput
    result['numHidden1'] = numHidden1
    result['numHidden2'] = numHidden2
    result['numOutput'] = numOutput
    result['params'] = params
    result['alpha'] = alpha_update

    save_obj(result, str(os.path.basename(__file__)))
    

