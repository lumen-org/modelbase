# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:43:29 2016

@author: philipp
"""

import numpy as np
from numpy import pi, exp, matrix, ix_


'''class MultiVariateGaussian ():
    def __init__(self, mu = 0, S = 0):
        self.mu = mu
        self.S = S
        
    def d (v): 
        return 

def mvGaussian (mu, S, )'''
   

   
    
class Model:
       
    def __init__ (self, name, data):
            self.name = name
            self.data = data
            
    def train (self):
        pass        
            
    def marginalize (self, keep = [], remove = []):
        if keep:
            Model._marginalize(keep)
        else:
            raise NotImplementedError()    
    
    def _marginalize (self, keep):
        pass
    
    def condition (self, pairs):
        pass
    
    def aggregate (self, method):
        pass
    
    def copy(self):
        pass


class MultiVariateGaussianModel (Model):
    
    def __init__ (self, name = "foo", data = []):
        # make sure these are matrix types (numpy.matrix)
        #self._mu = 0
        #self._S = 0                
        super().__init__(name, data)
        
        self._mu = matrix('1; 2')
        self._S = matrix('1 0.1; 0.4 2')        
        self._update()               
        
    def _update (self):
        self._n = self._mu.shape[0]        
        self._detS = np.linalg.det(self._S)        
        self._SInv = self._S.I
        
    def condition (self, pairs):
        raise NotImplementedError()
        # update sigma and mu according to GM script
        self._S = MultiVariateGaussianModel.UpperSchurCompl(self._S, ...)
        self._mu = ...
        self._update()
    
    def _marginalize (self, keep):
        # just select the part of mu and sigma that remains
        self._mu = self._mu[keep]  
        self._S = self._S[np.ix_(keep, keep)]
        self._update()
    
    def _density (self, x):   
        xmu = x - self._mu
        return (2*pi)**(-self._n/2) * self._detS**-.5 * exp( -.5 * xmu.T * self._SInv * xmu )
             
    def UpperSchurCompl (M, idx):
        '''Returns the upper Schur complement of matrix M with the 'upper block' indexed by i'''
        # derive correct index lists (using set opreations)
        i = idx
        j = list( set(range(0, M.shape[0])) - set(i) )
        # that the definition of the upper Schur complement
        return M[ix_(i,i)] - M[ix_(i,j)] * M[ix_(j,j)].I * M[ix_(j,i)]

#class RandomVariable:

    #def __init__             