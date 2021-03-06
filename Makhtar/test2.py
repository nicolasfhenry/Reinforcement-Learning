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
    env=gym.make('Acrobot-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    return env, obs_dim , act_dim


    
def episodeRoute(rnn, env, observation, steps=300):
    rewards = []
    inputs = env.reset()
    cum_reward = 0.0
    for j in range(steps):
        #env.render()
        #print(observation)
        outputs = rnn.feedForward(inputs)
        action = np.argmax(outputs)
        #print(action)
        inputs, reward, done, info = env.step(action)
        if done:
            break
        cum_reward += reward
        #print(reward)
        rewards.append(cum_reward) #useless, can be interesting if we want to plot the evolution of cum reward   
    return cum_reward

def runNN(rnn, env):
    initial_observation = env.reset()
    #observation=[ 0.99575298 , 0.09206522, 0.99953504,  0.03049091, -0.00171904, -0.08695283]
    print(observation)
    
    for t in range(1000):
        env.render()
<<<<<<< HEAD
        print(observation)
        
        outputs = rnn.feedForward(observation)  
        #print(rnn.feedForward(observation))
=======
        #print(observation)
        
        outputs = rnn.feedForward(observation)  
        print(rnn.feedForward(observation))
>>>>>>> 4e95a0ce050e035329af7468bf6f426b32805d9f
        
        action = np.argmax(outputs)
        print(action)
        observation, reward, done, info = env.step(action)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    
def fitness_shaping(rewards):
    utility=[(10*c+1)/(10* len(rewards)) for c in range(len(rewards))]
    
    return utility
    
def fitness_shaping_cheat(rewards):
    for i in range(len(rewards)):
        if rewards[i]==min(rewards):
            utility.append(0.01)
        else :
            utility.append((10*i+1)/(10* len(rewards)))
    
    return utility


def fitness_shaping_mountainCar(rewards):
    utility=[3*c+1 for c in range(len(rewards))]
    
    return utility


if __name__ == "__main__":
    #General parameters
    
    env, num_obs, num_action=initGym()
<<<<<<< HEAD
    num_episodes=50
    reward_episode=[]
    alpha =0.0001 #parameter gradient
    sigma=1 #parameter noise -update Fi
    num_workers=100
=======
    num_episodes=5
    reward_episode=[]
    alpha =0.1 #parameter gradient
    sigma=0.01 #parameter noise -update Fi
    num_workers=50
>>>>>>> 4e95a0ce050e035329af7468bf6f426b32805d9f

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
    dim_hidden_output=NN[0].hidden*NN[0].output
    dim_input_hidden=NN[0].input*NN[0].hidden
                    
    epsilon_wo=np.array([np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((NN[0].hidden,NN[0].output)) for x in range(num_workers)])
    epsilon_wi=np.array([np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((NN[0].input,NN[0].hidden)) for x in range(num_workers)])
    random_eps_wo=[np.random.randint(0,high=dim_input_hidden) for x in range(num_episodes)]
    random_eps_wi=[np.random.randint(0,high=dim_hidden_output) for x in range(num_episodes)]
    
    
        
    for i in range (num_episodes):
        
        epsilon_wo=np.array([ 0.5*(x+epsilon_wo[random_eps_wo[i]]) for x in epsilon_wo])
        epsilon_wi=np.array([ 0.5*(x+epsilon_wi[random_eps_wi[i]])  for x in epsilon_wi])
        
        print('episode : ',i)
        initial_observation=env.reset()
        #initial_observation=[ 0.99575298 , 0.09206522, 0.99953504,  0.03049091, -0.00171904, -0.08695283]
    
        reward_workers=[]
        incremental_gradient_wo=0
        incremental_gradient_wi=0

        list_reward_worker=[]        
        for worker in range(num_workers):
<<<<<<< HEAD
            #print('worker n°',worker)
            NN[worker].wo=NN[worker].wo+epsilon_wo[worker]*sigma #remark:we should merge the two, and reshape the matrix
            NN[worker].wi=NN[worker].wi+epsilon_wi[worker]*sigma
=======
            print('worker n°',worker)
            dim_hidden_output=NN[0].hidden*NN[0].output
            epsilon_wo=np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((NN[0].hidden,NN[0].output))
            dim_input_hidden=NN[0].input*NN[0].hidden
            epsilon_wi=np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((NN[0].input,NN[0].hidden))
            NN[worker].wo=NN[worker].wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
            NN[worker].wi=NN[worker].wi+epsilon_wi*sigma
>>>>>>> 4e95a0ce050e035329af7468bf6f426b32805d9f
            reward_worker=episodeRoute(NN[worker],env,initial_observation)
            list_reward_worker.append(reward_worker)
            #incremental_gradient_wo+=reward_worker*epsilon_wo #same !
            #incremental_gradient_wi+=reward_worker*epsilon_wi
     
        reward_episode.append([np.mean(list_reward_worker),np.median(list_reward_worker)])
        fitness=fitness_shaping(list_reward_worker)
        list_reward_worker=np.sort(list_reward_worker)
        
        #Formula to modify if we use firnaess shaping 
        incremental_gradient_wo=sum([x*y*z for (x,y,z) in zip(list_reward_worker,epsilon_wo,fitness)])
        incremental_gradient_wi=sum([x*y*z for (x,y,z) in zip(list_reward_worker,epsilon_wi,fitness)])
        
        for worker in range(num_workers):
            NN[worker].wo=NN[worker].wo-epsilon_wo[worker]*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wo
            NN[worker].wi=NN[worker].wi-epsilon_wi[worker]*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wi
        
        
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
