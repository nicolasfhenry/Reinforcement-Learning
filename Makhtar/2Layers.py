# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:20:30 2017

@author: Makhtar Ba
"""

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
    def __init__(self, input, hidden1,hidden2, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        
        self.input = input + 1 # add 1 for bias node
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output = output
        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah1 = [1.0] * self.hidden1
        self.ah2 = [1.0] * self.hidden2
        self.ao = [1.0] * self.output
        
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden1) 
        self.wh = np.random.randn(self.hidden1, self.hidden2) 
        self.wo = np.random.randn(self.hidden2, self.output) 
        
    

    
    def feedForward(self, input):
        if len(input) != self.input-1:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = input[i]
        # hidden activations
        for j in range(self.hidden1):
            sum = np.dot(self.ai[0:self.input-1],[self.wi[i][j] for i in range(self.input-1)])
            
            self.ah1[j] = sigmoid(sum)
        
        for j in range(self.hidden2):
            sum = np.dot(self.ah1[0:self.hidden1-1],[self.wh[i][j] for i in range(self.hidden1-1)])
            
            self.ah2[j] = sigmoid(sum)

        # output activations
        for k in range(self.output):
            sum = np.dot(self.ah2[0:self.hidden2],[self.wo[i][k] for i in range(self.hidden2)])
            
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
    num_episodes=100
    reward_episode=[]
    alpha =0.1
    sigma=0.01
    num_workers=500

    #Initialization of the neural net for the swimmer game
    numInput=num_obs
    numOutput=num_action
    numHidden1=8
    numHidden2=8
    
    NN=[NeuralNetwork(numInput,numHidden1,numHidden2,numOutput) for x in range(num_workers)]
    params=[np.random.randn(NN[0].input,NN[0].hidden1),np.random.randn(NN[0].hidden1,NN[0].hidden2),np.random.randn(NN[0].hidden2,NN[0].output)]
    
    for nn in NN:
        nn.wi=params[0]
        nn.wh=params[1]
        nn.w0=params[2]
        
    for i in range (num_episodes):
        observation=env.reset()
        reward_workers=[]
        incremental_gradient_wo=0
        incremental_gradient_wi=0
        incremental_gradient_wh=0
        
        
        for worker in range(num_workers):
            epsilon_wo=np.random.multivariate_normal([0 for x in range(NN[0].hidden2*NN[0].output)],np.identity(NN[0].hidden2*NN[0].output)).reshape((NN[0].hidden2,NN[0].output))
            epsilon_wi=np.random.multivariate_normal([0 for x in range(NN[0].input*NN[0].hidden1)],np.identity(NN[0].input*NN[0].hidden1)).reshape((NN[0].input,NN[0].hidden1))
            epsilon_wh=np.random.multivariate_normal([0 for x in range(NN[0].hidden1*NN[0].hidden2)],np.identity(NN[0].hidden1*NN[0].hidden2)).reshape((NN[0].hidden1,NN[0].hidden2))
            
            NN[worker].wo=NN[worker].wo+epsilon_wo*sigma
            NN[worker].wi=NN[worker].wi+epsilon_wi*sigma
            NN[worker].wh=NN[worker].wh+epsilon_wh*sigma
            
            reward_worker=episodeRoute(NN[worker],env,observation)
            incremental_gradient_wo+=reward_worker*epsilon_wo
            incremental_gradient_wi+=reward_worker*epsilon_wi
            incremental_gradient_wh+=reward_worker*epsilon_wh
            reward_workers.append(episodeRoute(NN[worker],env,observation))
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        #Formula to modify if we put multiple chocs
        
        for worker in range(num_workers):
            NN[worker].wo=NN[worker].wo-epsilon_wo*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wo
            NN[worker].wi=NN[worker].wi-epsilon_wi*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wi
        
        
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
    



