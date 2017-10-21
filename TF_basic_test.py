# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:21:48 2017
https://stackoverflow.com/questions/43765841/training-a-tensorflow-neural-network-on-openai-observations

@author: vassily
"""
### Libraries ###
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

### Parameters ###

gameName='MountainCar-v0'
alpha =0.01 #parameter gradient
sigma=0.01 #parameter noise -update Fi
num_workers=200
num_episodes=20
numHidden=8 # 8 neurons per Hidden layer

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

### Class Neural Network ###

class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: number of input neurons
        :param hidden_size: number of hidden neurons
        :param output_size: number of output neurons
        """        
        
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size       
       
         # Symbols
        self.X = tf.placeholder("float", shape=[self.input_size])
            
        # Weight initializations
        self.wi = tf.random_normal((self.input_size,self.hidden_size), stddev=0.1, dtype=tf.float32)
        self.wo = tf.random_normal((self.hidden_size,self.output_size), stddev=0.1, dtype=tf.float32)   
        
    
    def feedForward(self):
        self.wh = tf.nn.sigmoid(tf.matmul(tf.expand_dims(self.X, 1), self.wi, transpose_a=True))  # The \sigma function
        self.output = tf.matmul(self.wh, self.wo)  # The \varphi function
        self.predict = tf.argmax(self.output, axis=1)
        return(self.predict)

#Initialise OpenAI gym environment
def initGym():
    env=gym.make(gameName)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n    
    return env, obs_dim , act_dim


env, numInput, numOutput=initGym()
reward_episode=[]
initial_observation = env.reset()
observation=initial_observation


# Create NN object:
NN=NeuralNetwork(numInput,numHidden,numOutput)


def runNN(rnn, env):
    action=rnn.feedForward()  
    observation = env.reset()    
    with tf.Session():
        for step in range(100):
            env.render()
            evalAction=action.eval(feed_dict={NN.X: observation})[0]
            observation, reward, done, info = env.step(evalAction)
            print(observation)  
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

#runNN(NN,env)


def getRewardEpisode(rnn, env, NbSteps=1000):
    action=rnn.feedForward()  
    observation = env.reset()
    cum_reward = 0.0
    with tf.Session():
        for j in range(NbSteps):
            evalAction=action.eval(feed_dict={rnn.X: observation})[0]
            observation, reward, done, info = env.step(evalAction)
            if done:
                break
            cum_reward += reward        
        return cum_reward


NN=NeuralNetwork(numInput,numHidden,numOutput)
getRewardEpisode(NN,env,1000)


### OLD >>
def NN(myInput,numInput,numHidden,numOutput):  
    
    wi = tf.random_normal((numInput,numHidden), stddev=0.1, dtype=tf.float32)
    wo = tf.random_normal((numHidden,numOutput), stddev=0.1, dtype=tf.float32)    
    h = tf.nn.sigmoid(tf.matmul(tf.expand_dims(myInput, 1), wi, transpose_a=True))  # The \sigma function
    output = tf.matmul(h, wo)  # The \varphi function
    action = tf.argmax(output, axis=1)    
    return(action)
    
    

    

initial_observation = env.reset()
observation=initial_observation

myInput = tf.placeholder("float", shape=[numInput])
action=NN(myInput,numInput,numHidden,numOutput)

# Intialize the Session
with tf.Session():
    for step in range(100):
        env.render()
        evalAction=action.eval(feed_dict={myInput: observation})[0]
        observation, reward, done, info = env.step(evalAction)
        print(observation)
#print(sess.run(observation, feed_dict={myInput: initial_observation}))


# Close the session





