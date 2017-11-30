# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:08:12 2017

@author: Makhtar Ba
"""

import pickle
import matplotlib.pyplot
from neuralnets import NeuralNetwork
from useful_func import *

import os

os.chdir('C:/Users/Makhtar Ba/Documents/GitHub/Reinforcement-Learning')

reward_episode_GS_SR_1=load_obj('reward_episode_GS_semi')
reward_episode_GS_SR_2=load_obj('reward_episode_GS_semi_robust(2)')
reward_episode_GS_SR_3=load_obj('reward_episode_GS_semi_robust2(2)')


reward_episode_wo_GS_SR_1=load_obj('reward_episode_woGS_semi(2)')
reward_episode_wo_GS_SR_2=load_obj('reward_episode_woGS_semi_robust(2)')
reward_episode_wo_GS_SR_3=load_obj('reward_episode_woGS_semi_robust2(2)')



plt.plot([x[0] for x in reward_episode_GS_SR_1])
plt.plot([x[0] for x in reward_episode_GS_SR_2])
plt.plot([x[0] for x in reward_episode_GS_SR_3])



plt.plot([x[0] for x in reward_episode_wo_GS_SR_1])
plt.plot([x[0] for x in reward_episode_wo_GS_SR_2])
plt.plot([x[0] for x in reward_episode_wo_GS_SR_3])



params_GS_1=load_obj('params_GS_semi')
params_GS_2=load_obj('params_GS_semi_robust(2)')
params_GS_3=load_obj('params_GS_semi_robust2(2)')

params_wo_GS_1=load_obj('params_woGS_semi(2)')
params_wo_GS_2=load_obj('params_woGS_semi_robust(2)')
params_wo_GS_3=load_obj('params_woGS_semi_robust2(2)')

env, obs_dim , act_dim=initGym()
NN_GS_1=NeuralNetwork(obs_dim,8,act_dim)
NN_GS_2=NeuralNetwork(obs_dim,8,act_dim)
NN_GS_3=NeuralNetwork(obs_dim,8,act_dim)

simulation_reward_GS=[]

NN_GS_3.wi=params_GS_3[0]
NN_GS_3.wo=params_GS_3[1]

NN_GS_1.wi=params_GS_1[0]
NN_GS_1.wo=params_GS_1[1]

NN_GS_2.wi=params_GS_2[0]
NN_GS_2.wo=params_GS_2[1]


reward_episode_GS_1=[]
reward_episode_GS_2=[]
reward_episode_GS_3=[]

reward_episode_wo_GS_1=[]
reward_episode_wo_GS_2=[]
reward_episode_wo_GS_3=[]

initial_observation=0

for i in range(1000):
    reward_episode_GS_1.append(episodeRoute(NN_GS_1,env,initial_observation,steps=250))    
    reward_episode_GS_2.append(episodeRoute(NN_GS_2,env,initial_observation,steps=250))    
    reward_episode_GS_3.append(episodeRoute(NN_GS_3,env,initial_observation,steps=250))    

np.mean(reward_episode_GS_1)
np.mean(reward_episode_GS_2)
np.mean(reward_episode_GS_3)

np.std(reward_episode_GS_1)
np.std(reward_episode_GS_2)
np.std(reward_episode_GS_3)

plt.plot(reward_episode_GS_1)
plt.plot(reward_episode_GS_2)
plt.plot(reward_episode_GS_3)

np.mean(reward_episode_GS_1)/np.std(reward_episode_GS_1)
np.mean(reward_episode_GS_3)/np.std(reward_episode_GS_3)


NN_wo_GS_1=NeuralNetwork(obs_dim,8,act_dim)
NN_wo_GS_2=NeuralNetwork(obs_dim,8,act_dim)
NN_wo_GS_3=NeuralNetwork(obs_dim,8,act_dim)

simulation_reward_wo_GS=[]

NN_wo_GS_3.wi=params_wo_GS[0]
NN_wo_GS_3.wo=params_wo_GS[1]

NN_wo_GS_1.wi=params_wo_GS[0]
NN_wo_GS_1.wo=params_wo_GS[1]

NN_wo_GS_2.wi=params_wo_GS[0]
NN_wo_GS_2.wo=params_wo_GS[1]


reward_episode_wo_GS_1=[]
reward_episode_wo_GS_2=[]
reward_episode_wo_GS_3=[]



for i in range(1000):    
    reward_episode_wo_GS_1.append(episodeRoute(NN_wo_GS_1,env,initial_observation,steps=250))
    reward_episode_wo_GS_2.append(episodeRoute(NN_wo_GS_2,env,initial_observation,steps=250))
    reward_episode_wo_GS_3.append(episodeRoute(NN_wo_GS_3,env,initial_observation,steps=250))
 
np.mean(reward_episode_wo_GS_1)
np.mean(reward_episode_wo_GS_2)
np.mean(reward_episode_wo_GS_3)

np.std(reward_episode_wo_GS_1)
np.std(reward_episode_wo_GS_2)
np.std(reward_episode_wo_GS_3)
    
np.mean(reward_episode_wo_GS_1)/np.std(reward_episode_wo_GS_1)
np.mean(reward_episode_wo_GS_2)/np.std(reward_episode_wo_GS_2)
np.mean(reward_episode_wo_GS_3)/np.std(reward_episode_wo_GS_2)
   
plt.plot(reward_episode_wo_GS_1)
plt.plot(reward_episode_wo_GS_2)
plt.plot(reward_episode_wo_GS_3)




plt.plot([x[0] for x in reward_episode_wo_GS_SR_3])
plt.plot([x[0] for x in reward_episode_GS_SR_3])
 