# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:26:16 2017

@author: nicolas
"""

import numpy as np
from useful_func import *
import math 

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
        self.ai = [0.0] * self.input
        self.ah = [0.0] * self.hidden
        self.ao = [0.0] * self.output
        
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden) *0.01
        self.wo = np.random.randn(self.hidden, self.output)*0.01
        
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))


    
    def feedForward(self, input):
        if len(input) != self.input:
            raise ValueError('Wrong number of inputs!')
        # input activations
        self.ai = input
        # hidden activations
        self.ah=np.dot(np.transpose(self.wi),self.ai)
        self.ah = sigmoid(self.ah)
        # output activations
        self.ao=np.dot(np.transpose(self.wo),self.ah)
        self.ao = sigmoid(self.ao)
        
        
        return self.ao
    
    


class NeuralNetwork_2(object):
    def __init__(self, inputs, hidden1,hidden2, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        
        self.input = inputs+1 # add 1 for bias node
        self.hidden1 = hidden1+1
        self.hidden2 = hidden2+1
        self.output = output
        # set up array of 1s for activations
        self.ai = [0.0] * self.input
        self.ah1 = [0.0] * self.hidden1
        self.ah2 = [0.0] * self.hidden2
        self.ao = [0.0] * self.output
        
        self.wi = np.random.randn(self.input, self.hidden1) * 0.01
        self.wh = np.random.randn(self.hidden1, self.hidden2) * 0.01
        self.wo = np.random.randn(self.hidden2, self.output) * 0.01
        
        
    def feedForward(self, inputs):
         # create rasndomized weights
        
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs!')
        # input activations
        
        self.ai[:-1]=inputs# -1 is to avoid the bias
        
        # hidden 1 activations
        self.ah1[:-1]=np.dot(np.transpose(self.wi[:,:-1]),self.ai)
        self.ah1[:-1]=sigmoid(self.ah1[:-1])
        
        # hidden 2 activations
        
        self.ah2[:-1]=np.dot(np.transpose(self.wh[:,:-1]),self.ah1)
        self.ah2[:-1]=sigmoid(self.ah2[:-1])
        
        # output activations
        
        self.ao=np.dot(np.transpose(self.wo),self.ah2)
        
        return self.ao
    

class NeuralNetwork_2_nobias(object):
    def __init__(self, inputs, hidden1,hidden2, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        
        self.input = inputs # add 1 for bias node
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output = output
        # set up array of 1s for activations
        self.ai = [0.0] * self.input
        self.ah1 = [0.0] * self.hidden1
        self.ah2 = [0.0] * self.hidden2
        self.ao = [0.0] * self.output
        '''
        self.wi = np.random.randn(self.input, self.hidden1) 
        self.wh = np.random.randn(self.hidden1, self.hidden2) 
        self.wo = np.random.randn(self.hidden2, self.output) 
        '''
        self.wi = np.zeros((self.input, self.hidden1)) 
        self.wh = np.zeros((self.hidden1, self.hidden2)) 
        self.wo = np.zeros((self.hidden2, self.output))/ 
        
        
    def feedForward(self, inputs):
         # create rasndomized weights
        
        # input activations
        
        self.ai=inputs# -1 is to avoid the bias
        
        # hidden 1 activations
        #self.ah1=sigmoid(self.ah1)
        self.ah1=np.dot(np.transpose(self.wi),self.ai)
        self.ah1=np.tanh(self.ah1)
        # hidden 2 activations
        
        self.ah2=np.dot(np.transpose(self.wh),self.ah1)
        #self.ah2=sigmoid(self.ah2)
        self.ah2=np.tanh(self.ah2)
        
        # output activations
        
        self.ao=np.dot(np.transpose(self.wo),self.ah2)
        
        return self.ao
            
