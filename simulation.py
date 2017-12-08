# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:21:13 2017

@author: mmb2281
"""

import gym
import pickle 
from useful_func_mujoco import *
from neuralnets import *
from scipy.linalg import toeplitz


if __name__ == "__main__":
    env, num_obs, num_action = initGym()
    numInput=num_obs 
    numOutput=num_action
    numHidden1=num_obs*10 # 8 neurons per Hidden layer
    numHidden2=num_obs*10
    params=load_obj('params-v2Layers_50_0.01_0.08_222.110118942_254.805013573_+')
    NN = NeuralNetwork_2_nobias(numInput,numHidden1,numHidden2,numOutput)
    
    NN.wi=toeplitz(params[0][:numInput],params[0][numInput:])
    NN.wh=toeplitz(params[1][:numHidden1],params[1][numHidden1:])
    NN.wo=toeplitz(params[2][:numHidden2],params[2][numHidden2:])
    
    ''' 
    Whatch out to the value of the seed when running simulations 
    
    '''
    
    print(env)
    
    
    initial_observation=env.reset()    
    simulation_rewards=[]
    for i in range(100):
        seed=np.random.randint(1000,size=1)[0]
        env.seed(seed)
        simulation_rewards.append(episodeRoute(NN,env,initial_observation,steps=1000))
    save_obj(simulation_rewards,'simulation_params-v2Layers_50_0.01_0.08_222.110118942_254.805013573_+_shuffle_20')     
    #plt.plot(simulation_rewards)
    #runNN(NN, env) 
    
    
