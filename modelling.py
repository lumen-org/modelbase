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
   
def invertedIdxList (idx, len) :
    return list( set(range(0, len)) - set(idx) )   
    
def UpperSchurCompl (M, idx):
    '''Returns the upper Schur complement of matrix M with the 'upper block' indexed by i'''
    # derive correct index lists (using set opreations)
    i = idx
    j = invertedIdxList(i, M.shape[0])

    # that the definition of the upper Schur complement
    return M[ix_(i,i)] - M[ix_(i,j)] * M[ix_(j,j)].I * M[ix_(j,i)]    
    
class Model:
       
    def __init__ (self, name, data):
            self.name = name
            self.data = data
            self._aggrMethods = {}
            
    def train (self):
        pass        
            
    def marginalize (self, keep = [], remove = []):
        if keep:
            self._marginalize(keep)
        else:
            raise NotImplementedError()    
    
    def _marginalize (self, keep):
        pass
    
    def condition (self, pairs):
        pass
    
    def aggregate (self, method):
        if (method in self._aggrMethods):
            return self._aggrMethods[method]()
        else:
            raise NotImplementedError()
    
    def copy(self):
        pass


class MultiVariateGaussianModel (Model):
    
    def __init__ (self, name = "foo", data = []):
        # make sure these are matrix types (numpy.matrix)
        #self._mu = 0
        #self._S = 0                
        super().__init__(name, data)
        
        self._mu = matrix('1  2 .5').T
        self._S = matrix(
            '1   0.1 0   ;\
             0.1 0.3 0.6 ;\
             0   0.6 0.2 ')        
        self._update()
        
        self._aggrMethods = {
            'argmax': self._argmax,
            'argavg': self._argmax
        }
        
    def summary (self):
        return( "Multivariate Gaussian Model '" + self.name + "':\n" + \
                "dimension:\n" + str(self._n) + "\n" + \
                "mu:\n" + str(self._mu) + "\n" + \
                "sigma:\n" + str(self._S) + "\n" )
        
    def _update (self):
        self._n = self._mu.shape[0]        
        self._detS = np.linalg.det(self._S)        
        self._SInv = self._S.I
        
    def condition (self, pairs):        
        i, xj = zip(*pairs)
        j = invertedIdxList(i, self._n)
        
        # store old sigma and mu
        S = self._S
        mu = self._mu
        
        # update sigma and mu according to GM script
        self._S = MultiVariateGaussianModel.UpperSchurCompl(S, i)        
        self._mu = mu[i] + S[ix_(i,j)] * S[ix_(j,j)].I * (xj - mu[j])
        self._update()
    
    def _marginalize (self, keep):
        # just select the part of mu and sigma that remains
        self._mu = self._mu[keep]  
        self._S = self._S[np.ix_(keep, keep)]
        self._update()
    
    def _density (self, x):   
        xmu = x - self._mu
        return (2*pi)**(-self._n/2) * self._detS**-.5 * exp( -.5 * xmu.T * self._SInv * xmu )
        
    def _argmax (self):
        return self._mu
        
    def copy (self):
        mycopy = MultiVariateGaussianModel(name = self.name, data = self.data)
        mycopy._mu = self._mu
        mycopy._S = self._S
        mycopy._update()
        return mycopy