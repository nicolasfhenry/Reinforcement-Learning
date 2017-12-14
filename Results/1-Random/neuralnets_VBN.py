# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:26:16 2017

@author: Vassily
"""

import numpy as np


class NeuralNetwork(object):
    def __init__(self, dim_input, dim_hidden_1, dim_hidden_2, dim_output, VBN_dict):
        """
        :param dim_input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.use_bias=False
        self.use_VBN=True
        
        self.n_x = dim_input
        self.n_h1 = dim_hidden_1
        self.n_h2 = dim_hidden_2
        self.n_y = dim_output
        
        if (self.use_bias==True):
            self.b1 = np.zeros((self.n_h1,1))
            self.b2 = np.zeros((self.n_h2,1))     

        if (self.use_bias==True):
            self.beta1 = np.zeros((self.n_h1,1))
            self.beta2 = np.zeros((self.n_h2,1))
            self.gamma1 = np.ones((self.n_h1,1))
            self.gamma2 = np.ones((self.n_h2,1))   
        
        
        
        #self.W1 = np.random.randn(self.n_h1,self.n_x) * 0.01
        #self.W2 = np.random.randn(self.n_h2,self.n_h1) * 0.01
        #self.W3 = np.random.randn(self.n_y,self.n_h2) * 0.01
        self.W1 = np.zeros((self.n_h1,self.n_x))
        self.W2 = np.zeros((self.n_h2,self.n_h1))     
        self.W3 = np.zeros((self.n_y,self.n_h2))
        
        
        self.Ai = np.zeros((self.n_x,1))
        self.A1 = np.zeros((self.n_h1,1))
        self.A2 = np.zeros((self.n_h2,1))
        self.Ao = np.zeros((self.n_y,1))
                
        self.Z1 = np.zeros((self.n_h1,1))
        self.Z2 = np.zeros((self.n_h2,1))
        self.Zo = np.zeros((self.n_y,1))
        
        
        #create VBN statistics:
        #self.mu_i = VBN_dict['mu_i']
        #self.var_i = VBN_dict['var_i']
        self.mu_h1 = VBN_dict['mu_h1'].reshape((self.n_h1,1))
        self.var_h1 = VBN_dict['var_h1'].reshape((self.n_h1,1))
        self.mu_h2 = VBN_dict['mu_h2'].reshape((self.n_h2,1))
        self.var_h2 = VBN_dict['var_h2'].reshape((self.n_h2,1))
        #self.mu_o = VBN_dict['mu_o']
        #self.var_o = VBN_dict['var_o']
        


    def feedForward(self,X_old):
        
        X=X_old.reshape((self.n_x,1))
    
        assert (X.shape == (self.n_x,1))
        assert (self.mu_h1.shape == (self.n_h1,1))
        assert (self.var_h1.shape == (self.n_h1,1))
        assert (self.mu_h2.shape == (self.n_h2,1))
        assert (self.var_h2.shape == (self.n_h2,1))
        
        if(self.use_VBN ==True):
            #print('with VBN')
            return(self.feedForward_VBN(X))
        else:
            if (self.use_bias==True):
                #print('with bias')
                return(self.feedForward_withBias(X))
            else:
                #print('without bias')
                return(self.feedForward_noBias(X))
            
    
    def feedForward_noBias(self, X):
        self.Ai=X
        self.Z1 = np.dot(self.W1,self.Ai)
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.W2,self.A1)
        self.A2 = np.tanh(self.Z2)
        self.Zo = np.dot(self.W3,self.A2)
        self.Ao= self.sigmoid(self.Zo)
        return(self.Ao)
        
    def feedForward_withBias(self, X):
        self.Ai= X
        self.Z1 = np.dot(self.W1,self.Ai)+self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.W2,self.A1)+self.b2
        self.A2 = np.tanh(self.Z2)
        self.Zo = np.dot(self.W3,self.A2)
        self.Ao= self.sigmoid(self.Zo)
        return(self.Ao)
        
    
    def feedForward_VBN(self, X):
        self.Ai=X
        self.Z1 = np.dot(self.W1,self.Ai)
        Z1_VBN = self.applyVBN(self.Z1, self.mu_h1, self.var_h1, self.gamma1, self.beta1)
        self.A1 = np.tanh(Z1_VBN)
        self.Z2 = np.dot(self.W2,self.A1)
        Z2_VBN = self.applyVBN(self.Z2, self.mu_h2, self.var_h2, self.gamma2, self.beta2)
        self.A2 = np.tanh(Z2_VBN)
        self.Zo = np.dot(self.W3,self.A2)
        self.Ao= self.sigmoid(self.Zo)
        return(self.Ao)
        

        
        
    def applyVBN(self, X, mu, var, gamma, beta):        
        X_norm = (X-mu)/np.sqrt(var + 1e-8)
        return(np.multiply(gamma,X_norm) + beta)
        
    def sigmoid(self,z):
         s = 1/(1+np.exp(-z))  
         return s