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
alpha =0.5 #parameter gradient
sigma=0.01 #parameter noise -update Fi
num_workers=50
num_episodes=4
numHidden=8 # 8 neurons per Hidden layer

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

### Class Neural Network ###

class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size,wi,wo):
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
        
        self.wi=wi
        self.wo=wo
        
        '''
        self.wi = tf.random_normal((self.input_size,self.hidden_size), stddev=0.1, dtype=tf.float32)
        self.wo = tf.random_normal((self.hidden_size,self.output_size), stddev=0.1, dtype=tf.float32)   
        '''
    
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


### MAIN ###

if __name__ == "__main__":
    #General parameters

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
    
    
    # Create a NN object for each worker:
       
    params=[tf.random_normal((numInput,numHidden), stddev=0.1, dtype=tf.float32),tf.random_normal((numHidden,numOutput), stddev=0.1, dtype=tf.float32)]
    NN=[NeuralNetwork(numInput,numHidden,numOutput,params[0],params[1]) for x in range(num_workers)]
    
    
        
    #Simulations
    
    for i in range (num_episodes):
        print('episode : ',i)
        reward_workers=[]
        incremental_gradient_wo=0
        incremental_gradient_wi=0
        '''
        epsilon_wo_matrix=[np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((NN[0].hidden_size,NN[0].output_size)) for x in range(num_workers)]
        epsilon_wi=[np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((NN[0].input_size,NN[0].hidden_size)) for x in range(num_workers)]
        '''    
        for worker in range(num_workers):
            print('worker n°',worker)
            dim_hidden_output=NN[0].hidden_size*NN[0].output_size
            dim_input_hidden=NN[0].input_size*NN[0].hidden_size
            epsilon_wo=np.random.multivariate_normal([0 for x in range(dim_hidden_output)],np.identity(dim_hidden_output)).reshape((NN[0].hidden_size,NN[0].output_size))
            epsilon_wi=np.random.multivariate_normal([0 for x in range(dim_input_hidden)],np.identity(dim_input_hidden)).reshape((NN[0].input_size,NN[0].hidden_size))
            NN[worker].wo=NN[worker].wo+epsilon_wo *sigma #remark:we should merge the two, and reshape the matrix
            NN[worker].wi=NN[worker].wi+epsilon_wi*sigma
            reward_worker=getRewardEpisode(NN[worker],env)
            reward_workers.append(reward_worker)
            incremental_gradient_wo+=reward_worker*epsilon_wo #same !
            incremental_gradient_wi+=reward_worker*epsilon_wi
            
        
        reward_episode.append([np.mean(reward_workers),np.median(reward_workers)])
        
        #Formula to modify if we put multiple chocs
        
        for worker in range(num_workers):
            NN[worker].wo=NN[worker].wo-epsilon_wo*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wo
            NN[worker].wi=NN[worker].wi-epsilon_wi*sigma+alpha*1/(num_workers*sigma)*incremental_gradient_wi
        
        
    print(reward_episode)   
    plt.plot([x[0] for x in reward_episode])
    runNN(NN[1], env)
