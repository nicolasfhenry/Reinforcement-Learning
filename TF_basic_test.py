# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:21:48 2017
https://stackoverflow.com/questions/43765841/training-a-tensorflow-neural-network-on-openai-observations

@author: vassily
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def initGym():
    env=gym.make('MountainCar-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    return env, obs_dim , act_dim


env, num_obs, num_action=initGym()
num_episodes=20
reward_episode=[]
alpha =0.01 #parameter gradient
sigma=0.01 #parameter noise -update Fi
num_workers=200

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer

def NN(myInput,numInput,numHidden,numOutput):  
    
    wi = tf.random_normal((numInput,numHidden), stddev=0.1, dtype=tf.float32)
    wo = tf.random_normal((numHidden,numOutput), stddev=0.1, dtype=tf.float32)    
    h = tf.nn.sigmoid(tf.matmul(tf.expand_dims(myInput, 1), wi, transpose_a=True))  # The \sigma function
    output = tf.matmul(h, wo)  # The \varphi function
    action = tf.argmax(output, axis=1)    
    return(action)
    

initial_observation = env.reset()
myInput = tf.placeholder("float", shape=[numInput])
action=NN(myInput,numInput,numHidden,numOutput)

# Intialize the Session
with tf.Session():
    action2=action.eval(feed_dict={myInput: initial_observation})
    print(action2)
    observation, reward, done, info = env.step(action2[0])
    print(observation)
#print(sess.run(observation, feed_dict={myInput: initial_observation}))


# Close the session





