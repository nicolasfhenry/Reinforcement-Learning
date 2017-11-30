# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:11:37 2017

@author: nicolas henry
"""

import tensorflow as tf
import numpy as np
import multiprocessing  
import gym

import matplotlib.pyplot as plt
import time
import math

#from neuralnets import NeuralNetwork
#from useful_func import *


#Params:

num_episodes = 25


#Acrobot
alphaValue = 0.1 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100
TFseed=42
tf.set_random_seed(TFseed)

################

def alpha(i, alphaValue):
    whenDecay=int(num_episodes/2)
    if (i<whenDecay):
        return(alphaValue)
    elif (i>=whenDecay)&(i<whenDecay*2-10):
        return(alphaValue*(1-(i-whenDecay)/whenDecay))
    else:
        return(alphaValue*(11/whenDecay))

#plt.plot([alpha(i, 0.1) for i in range(200)])



def initGym():
    env=gym.make('Acrobot-v1')
    #env=gym.make('CartPole-v0')
    #env=gym.make('MountainCar-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    return env, obs_dim , act_dim


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

def getRewardEpisode(rnn, env, NbSteps=300):
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
#getRewardEpisode(NN,env,1000)
        
        
def fitness_shaping_paper(rewards):
    length=len(rewards)
    temp=[max(0,(math.log(length/2+1)-math.log(length-(s)+1)))/(sum([max(0,math.log(length/2+1)-math.log(j+1)) for j in range(length)])) for s in range(len(rewards))]
    #temp=[x-1/length for x in temp]
    return temp


"""
#Mountain Cart
alpha = 10 #parameter gradient
sigma = 0.5 #parameter noise -update Fi
num_workers=100
################
"""

    
env, num_obs, num_action = initGym()
tf.reset_default_graph()

#Initialization of the neural net for the game
numInput=num_obs 
numOutput=num_action
numHidden=8 # 8 neurons per Hidden layer

dim_input_hidden=numInput*numHidden
dim_hidden_output=numHidden*numOutput




def worker(input_worker):
    """Explanations"""
    #Global variables:
    global numInput,numOutput,numHidden
    global dim_hidden_output, dim_hidden_output
    global sigma
    global env
    
    #Lovcal:
    
    seed = int(input_worker[0])
    p = input_worker[1]

    env.seed(0) 
    
    
    #Neural Networks:
    NN = NeuralNetwork(numInput,numHidden,numOutput)

    NN.wi=p[0]
    NN.wo=p[1]

    #distortions
    epsilon_wo = np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((numHidden,numOutput)).astype(np.float32)
    epsilon_wi = np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((numInput,numHidden)).astype(np.float32)
    #parameters update
    NN.wo=NN.wo+epsilon_wo*sigma #remark:we should merge the two, and reshape the matrix
    NN.wi=NN.wi+epsilon_wi*sigma
    
    #initial_observation=env.reset()
    reward_worker=getRewardEpisode(NN,env,300)
    
    #reward_worker=episodeRoute(NN,env,initial_observation,steps=250)
    
    
    return(reward_worker,epsilon_wi,epsilon_wo)




def main(seeds,params):
    '''
    input_worker = list(zip(seeds,[params]*len(seeds)))
    
    pool = multiprocessing.Pool(4)
    results = pool.map(worker,input_worker)

    pool.close()
    pool.join()
    '''

    results=[]
    for seed in seeds:
        input_worker=list([seed,params])  
        results.append(worker(input_worker))
        
    return(results)
    

if __name__ == "__main__":
    #General parameters
    params = [np.random.randn(numInput,numHidden).astype(np.float32),np.random.randn(numHidden,numOutput).astype(np.float32)]
     
    reward_episode=[]
    
    for i in range (num_episodes):
        print('episode : ',i)
        
        
        
        seeds = np.random.randint(10000,size=num_workers)
        
        reward_workers,epsilon_wi,epsilon_wo =  [list(x) for x in  zip(*main(seeds,params))]
        
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
        print(alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)]))

        params[0] = params[0] - alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wi,reward_workers,fitness)])
        params[1] = params[1] - alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F*w for eps,F,w in zip(epsilon_wo,reward_workers,fitness)])
        """
     
        #grad1:
        params[0] = params[0] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wi,reward_workers)])
        params[1] = params[1] + alpha(i,alphaValue)*(1/(num_workers*sigma))*sum([eps*F for eps,F in zip(epsilon_wo,reward_workers)])
        """
        
        
        #print(reward_episode[-1][0])
    print(reward_episode)   

    plt.plot([x[0] for x in reward_episode])
    #save_obj(params,'params-v2')
    ### Test:
    NN = NeuralNetwork(numInput,numHidden,numOutput)
    NN.wi=params[0]
    NN.wo=params[1]
    runNN(NN, env)    