# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:26:16 2017

@author: nicolas
"""

import numpy as np
from useful_func import *

class NeuralNetwork(object):
    def __init__(self, input, hidden, output, VBN_dict):
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
        
        #set up array of ls for weigthed sum
        self.zi = [1.0] * self.input
        self.zh = [1.0] * self.hidden
        self.zo = [1.0] * self.output
        
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden) 
        self.wo = np.random.randn(self.hidden, self.output) 
        
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden)) #useless ?
        self.co = np.zeros((self.hidden, self.output)) #useless ?
        
        #create VBN statistics:
        self.mu_i = VBN_dict['mu_i']
        self.var_i = VBN_dict['var_i']
        self.mu_h = VBN_dict['mu_h']
        self.var_h = VBN_dict['var_h']
        self.mu_o = VBN_dict['mu_o']
        self.var_o = VBN_dict['var_o']
     


    
    def feedForward(self, input):
        #to vectorize for speed up !!
        if len(input) != self.input:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input): # -1 is to avoid the bias
            self.ai[i] = input[i]
        # hidden activations
        for j in range(self.hidden):
            self.zh[j] = np.dot(self.ai[0:self.input],[self.wi[i][j] for i in range(self.input)])            
            self.ah[j] = sigmoid(self.zh[j])
        # output activations
        for k in range(self.output):            
            self.zo[k] = np.dot(self.ah[0:self.hidden],[self.wo[i][k] for i in range(self.hidden)])            
            self.ao[k] = sigmoid(self.zo[k])
        return self.ao
    
    def feedForward_VBN(self, input):
        #to vectorize for speed up !! actually more easy to apply VBN with vectorized implementation
        if len(input) != self.input:
            raise ValueError('Wrong number of inputs!')
        # input activations
        for i in range(self.input): # -1 is to avoid the bias
            self.ai[i] = input[i]
            # VBN here as well ??
        # hidden activations
        for j in range(self.hidden):
            self.zh[j] = np.dot(self.ai[0:self.input],[self.wi[i][j] for i in range(self.input)])
            #add VBN step here            
            self.ah[j] = sigmoid(self.zh[j])
        # output activations
        for k in range(self.output):            
            self.zo[k] = np.dot(self.ah[0:self.hidden],[self.wo[i][k] for i in range(self.hidden)])
            # question: should we put VBN here as well ?? check paper
            self.ao[k] = sigmoid(self.zo[k])
        return self.ao
        
        
    def applyVBN(X, mu, var, gamma, beta):    
        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        return(gamma*X_norm + beta)
        
    