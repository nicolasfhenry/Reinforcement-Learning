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
        self.ai = input
        # hidden activations
        self.ah=np.dot(np.transpose(self.wi),)
        for j in range(self.hidden):
            sumInput = np.dot(self.ai[0:self.input],[self.wi[i][j] for i in range(self.input)])            
            self.ah[j] = sigmoid(sumInput)
        # output activations
        for k in range(self.output):            
            sumHidden = np.dot(self.ah[0:self.hidden],[self.wo[i][k] for i in range(self.hidden)])            
            self.ao[k] = sigmoid(sumHidden)
        return self.ao
    
    


class NeuralNetwork_2(object):
    def __init__(self, inputs, hidden1,hidden2, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        
        self.input = inputs+1# add 1 for bias node
        self.hidden1 = hidden1+1
        self.hidden2 = hidden2+1
        self.output = output
        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah1 = [1.0] * self.hidden1
        self.ah2 = [1.0] * self.hidden2
        self.ao = [1.0] * self.output
        
       
        
    def feedForward(self, inputs):
         # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden1) 
        self.wh = np.random.randn(self.hidden1, self.hidden2) 
        self.wo = np.random.randn(self.hidden2, self.output) 
        
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input-1): # -1 is to avoid the bias
            self.ai[i] = inputs[i]
        # hidden activations
        for j in range(self.hidden1-1):
            sum = np.dot(self.ai,self.wi[:,j])
            
            #self.ah1[j] = sigmoid(sum)
            self.ah1[j] = math.tanh(sum)
            
        for j in range(self.hidden2-1):
            sum = np.dot(self.ah1,self.wh[:,j])
            
            #self.ah2[j] = sigmoid(sum)
            self.ah2[j] = math.tanh(sum)
            
        # output activations
        for k in range(self.output):
            
            sum_ = np.dot(self.ah2,self.wo[:,k] )
            
            self.ao[k] = sum_
        
        return self.ao
    
        
