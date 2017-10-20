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
        
        self.input = input # bias node 
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
        if len(input) != self.input:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input): # -1 is to avoid the bias
            self.ai[i] = input[i]
        # hidden activations
        for j in range(self.hidden):
            sumInput = np.dot(self.ai[0:self.input],[self.wi[i][j] for i in range(self.input)])            
            self.ah[j] = sigmoid(sumInput)
        # output activations
        for k in range(self.output):            
            sumHidden = np.dot(self.ah[0:self.hidden],[self.wo[i][k] for i in range(self.hidden)])            
            self.ao[k] = sigmoid(sumHidden)
        return self.ao
    



    
#Initialize the environment 

def initGym():
    env=gym.make('MountainCar-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    return env, obs_dim , act_dim


    
def episodeRoute(rnn, env, observation, steps=1000):
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
        rewards.append(cum_reward) #useless, can be interesting if we want to plot the evolution of cum reward   
    return cum_reward

def runNN(rnn, env):
    initial_observation = env.reset()
    observation=initial_observation
    for t in range(1000):
        env.render()
        #print(observation)
        outputs = rnn.feedForward(observation)  
        action = np.argmax(outputs)
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break



if __name__ == "__main__":
    #General parameters
    
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
    
    NN=[NeuralNetwork(numInput,numHidden,numOutput) for x in range(num_workers)]
    params=[np.random.randn(NN[0].input,NN[0].hidden),np.random.randn(NN[0].hidden,NN[0].output)]
    
    for nn in NN:
        nn.wi=params[0]
        nn.w0=params[1]
        
        
    #Simulations
    
    for i in range (num_episodes):
        print('episode : ',i)
        initial_observation=env.reset()
        reward_workers=[]
        incremental_gradient_wo=0
        incremental_gradient_wi=0
        
        for worker in range(num_workers):
            dim_hidden_output=NN[0].hidden*NN[0].output
            epsilon_wo=np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((NN[0].hidden,NN[0].output))
            dim_input_hidden=NN[0].input*NN[0].hidden
            epsilon_wi=np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((NN[0].input,NN[0].hidden))
            NN[worker].wo=NN[worker].wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
            NN[worker].wi=NN[worker].wi+epsilon_wi*sigma
            reward_worker=episodeRoute(NN[worker],env,initial_observation)
            incremental_gradient_wo+=reward_worker*epsilon_wo #same !
            incremental_gradient_wi+=reward_worker*epsilon_wi
            reward_workers.append(episodeRoute(NN[worker],env,initial_observation))
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        #Formula to modify if we put multiple chocs
        
        for worker in range(num_workers):
            NN[worker].wo=NN[worker].wo-epsilon_wo*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wo
            NN[worker].wi=NN[worker].wi-epsilon_wi*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wi
        
        
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
    runNN(NN[1], env)


