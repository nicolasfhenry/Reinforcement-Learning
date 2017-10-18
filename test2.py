# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:42:49 2017

@author: Makhtar Ba
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

# Network Construction using only numpy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y)), the way we use this y is already a sigmoid

def dsigmoid(y):
    return y * (1.0 - y)  


class NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output
        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden) 
        self.wo = np.random.randn(self.hidden, self.output) 
        
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))


    
    def feedForward(self, input):
        if len(input) != self.input-1:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = input[i]
        # hidden activations
        for j in range(self.hidden):
            sum = np.dot(self.ai[0:self.input-1],[self.wi[i][j] for i in range(self.input-1)])
            
            self.ah[j] = sigmoid(sum)
        # output activations
        for k in range(self.output):
            sum = 0.0
            sum = np.dot(self.ah[0:self.hidden],[self.wo[i][k] for i in range(self.hidden)])
            
            self.ao[k] = sigmoid(sum)
        return self.ao
    
    def update(self,new_param_i,new_param_o):
        self.wi=new_param_i
        self.wo=new_param_o
        


    
#Initialize the environment 

def initGym():
    env=gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    return env, obs_dim , act_dim


    
def episodeRoute(rnn, env, observation, steps=5000):
    rewards = []
    inputs = env.reset()
    cum_reward = 0.0
    for j in range(steps):
        #env.render()
        #print(observation)
        outputs = rnn.feedForward(inputs)
        action = np.argmax(outputs)
        inputs, reward, done, info = env.step(action)
        if done:
            break
        cum_reward += reward
    rewards.append(cum_reward)
    return np.sum(rewards)

if __name__ == "__main__":
    #General parameters
    
    env, num_obs, num_action=initGym()
    num_episodes=400
    reward_episode=[]
    alpha =0.01
    sigma=0.01
    num_workers=600

    #Initialization of the neural net for the swimmer game
    numInput=num_obs
    numOutput=num_action
    numHidden=8
    
    NN=[NeuralNetwork(numInput,numHidden,numOutput) for x in range(num_workers)]
    params=[np.random.randn(NN[0].input,NN[0].hidden),np.random.randn(NN[0].hidden,NN[0].output)]
    
    for nn in NN:
        nn.wi=params[0]
        nn.w0=params[1]
        
        
    #Simulations
    
    for i in range (num_episodes):
        observation=env.reset()
        reward_workers=[]
        incremental_gradient_wo=0
        incremental_gradient_wi=0
        
        for worker in range(num_workers):
            epsilon_wo=np.random.multivariate_normal([0 for x in range(NN[0].hidden*NN[0].output)],np.identity(NN[0].hidden*NN[0].output)).reshape((NN[0].hidden,NN[0].output))
            epsilon_wi=np.random.multivariate_normal([0 for x in range(NN[0].input*NN[0].hidden)],np.identity(NN[0].input*NN[0].hidden)).reshape((NN[0].input,NN[0].hidden))
            NN[worker].wo=NN[worker].wo+epsilon_wo*sigma
            NN[worker].wi=NN[worker].wi+epsilon_wi*sigma
            reward_worker=episodeRoute(NN[worker],env,observation)
            incremental_gradient_wo+=reward_worker*epsilon_wo
            incremental_gradient_wi+=reward_worker*epsilon_wi
            reward_workers.append(episodeRoute(NN[worker],env,observation))
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        #Formula to modify if we put multiple chocs
        
        for worker in range(num_workers):
            NN[worker].wo=NN[worker].wo-epsilon_wo*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wo
            NN[worker].wi=NN[worker].wi-epsilon_wi*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wi
        
        
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
    



