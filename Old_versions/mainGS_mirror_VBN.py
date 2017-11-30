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
import matplotlib.pyplot as plt
import time

from neuralnets_VBN import NeuralNetwork
from useful_func_mujoco import *


#Params:

env, num_obs, num_action = initGym()

num_episodes = 100


#Acrobot
alphaValue = 0.3 #parameter gradient
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

plt.plot([alpha(i, 0.3) for i in range(200)])

"""
#Mountain Cart
alpha = 10 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100

"""

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action

numHidden=13 # 8 neurons per Hidden layer

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
    NN = NeuralNetwork(numInput,numHidden,numOutput, VBN_dict)
    
    NN.wi=p[0]
    NN.wo=p[1]
    
    #distortions
    
    epsilon = np.random.multivariate_normal(np.zeros(dim_hidden_output+dim_input_hidden),np.identity(dim_hidden_output+dim_input_hidden))
    epsilon_wo= epsilon[0:dim_hidden_output].reshape((numHidden,numOutput))
    epsilon_wi= epsilon[dim_hidden_output:dim_hidden_output+dim_input_hidden].reshape((numInput,numHidden))

    
    #epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((numHidden,numOutput))
    #epsilon_wi = np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((numInput,numHidden))
    
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    #initial_observation=env.reset()
    
    reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    
    
    return(reward_worker,epsilon_wi,epsilon_wo)


def worker_train_VBN(input_worker_VBN):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global env
    
    
    #Local:

    seed=int(input_worker_VBN[0])
    p = input_worker_VBN[1]
    
    env.seed(seed) 
    #np.random.seed(seed)    
    VBN_dict = {}
    VBN_dict['mu_i']=0
    VBN_dict['var_i']=0
    VBN_dict['mu_h']=0
    VBN_dict['var_h']=0
    VBN_dict['mu_o']=0
    VBN_dict['var_o']=0    
    
    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput, VBN_dict)    
    NN.wi=p[0]
    NN.wo=p[1]
    
    steps=250
    ai = env.reset()  
    num_step=steps
    
    for j in range(steps):
        
        ao = NN.feedForward(ai)
        #to transfer to the main 
        # question: how many worker for this ?
        
        sum_zi=[0.] * numInput        
        sum_zh=[0.] * numHidden  
        sum_zo=[0.] *numOutput
        sum_zi2=[0.] * numInput        
        sum_zh2=[0.] * numHidden  
        sum_zo2=[0.] *numOutput
        
 
        
        sum_zi=[sum(x) for x in zip(sum_zi, NN.zi)]   ### VERY WEIRD !! ALWAYS EQUAL TO 1 ?
        sum_zh=[sum(x) for x in zip(sum_zh, NN.zh)]
        sum_zo=[sum(x) for x in zip(sum_zo, NN.zo)]
        
        sum_zi2=[sum(x) for x in zip(sum_zi2, square(NN.zi))]
        sum_zh2=[sum(x) for x in zip(sum_zh2, square(NN.zh))]
        sum_zo2=[sum(x) for x in zip(sum_zo2, square(NN.zo))]
        
        '''
        sum_zi=map(add, sum_zi, NN.zi)
        sum_zh=map(add, sum_zh, NN.zh)
        sum_zo=map(add, sum_zo, NN.zo)
        
        sum_zi=map(add, sum_zi2, square(NN.zi))
        sum_zh=map(add, sum_zh2, square(NN.zh))
        sum_zo=map(add, sum_zo2, square(NN.zo))
        '''
        
                
        action=np.argmax(ao)      
        ai, reward, done, info = env.step(action)
        
        if done:
            break
            num_step=j     
            
    return(sum_zi,sum_zh,sum_zo,sum_zi2,sum_zh2,sum_zo2,num_step)



def worker_GS(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global sigma
    global env
    global VBN_dict
    
    #Local:

    seed=int(input_worker[0])
    epsilon = input_worker[1]
    p = input_worker[2]
    
    env.seed(seed) 
    #np.random.seed(seed)    
    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput, VBN_dict)
    
    NN.wi=p[0]
    NN.wo=p[1]
    
    #distortions
    
    epsilon_wo= epsilon[0:dim_hidden_output].reshape((numHidden,numOutput))
    epsilon_wi= epsilon[dim_hidden_output:dim_hidden_output+dim_input_hidden].reshape((numInput,numHidden))

    
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
    
def computeVBN(seeds_VBN, params):
        
    input_worker_VBN = list(zip(seeds_VBN,[params]*len(seeds_VBN)))

    pool = multiprocessing.Pool(4)
    results = pool.map(worker_train_VBN,input_worker_VBN)

    pool.close()
    pool.join()
    
    return(results)

if __name__ == "__main__":
    #General parameters
    params = [np.random.randn(numInput,numHidden),np.random.randn(numHidden,numOutput)]  
    
    reward_episode=[]    
    for episode in range (num_episodes):
        
        print('episode : ',episode)
        
        print('Virtual Batch Normalization')
        ### if too slow, change everything into np.arrays
        num_VBN_workers=20
        seeds_VBN = np.random.randint(10000,size=num_VBN_workers)
        
        sum_zi,sum_zh,sum_zo,sum_zi2,sum_zh2,sum_zo2,num_step_list =  [list(x) for x in  zip(*computeVBN(seeds_VBN,params))]
        num_step=float(sum(num_step_list))
        normalizing_factor=1/num_step
       
        mu_zi=[normalizing_factor*sum(i) for i in zip(*sum_zi)]
        mu2_zi=[normalizing_factor*sum(i) for i in zip(*sum_zi2)]
        var_zi=[(i-j) for i,j in zip(mu2_zi, square(mu_zi))]
        
        mu_zh=[normalizing_factor*sum(i) for i in zip(*sum_zh)]
        mu2_zh=[normalizing_factor*sum(i) for i in zip(*sum_zh2)]
        var_zh=[i-j for i,j in zip(mu2_zh, square(mu_zh))]
        
        mu_zo=[normalizing_factor*sum(i) for i in zip(*sum_zo)]
        mu2_zo=[normalizing_factor*sum(i) for i in zip(*sum_zo2)]
        var_zo=[i-j for i,j in zip(mu2_zo, square(mu_zo))]
        
        #save VBN statistics:
        VBN_dict = {}
        VBN_dict['mu_i']=mu_zi
        VBN_dict['var_i']=var_zi
        VBN_dict['mu_h']=mu_zh
        VBN_dict['var_h']=var_zh
        VBN_dict['mu_o']=mu_zo
        VBN_dict['var_o']=var_zo
        # here we need to find a way to pass this argument in the workers, make it a global var ?         
                
        #print(mu_zi, var_zi)
        #print(mu_zh, var_zh)    
        #print(mu_zo, var_zo)        
        
        print('Episode Computations')        
        
        #Creating Epsilon
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

    plt.plot([x[1] for x in reward_episode])
    save_obj(params,'params-vGS-1')
    ### Test:

"""
NN = NeuralNetwork(numInput,numHidden,numOutput)
NN.wi=params[0]
NN.wo=params[1]
  
runNN(NN, env) 
"""