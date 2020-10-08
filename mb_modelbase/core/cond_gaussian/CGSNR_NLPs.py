# -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de)

"""



import numpy as np
from math import inf

############################################################################### GAUSS

class GMNLP_GAUSS_SNR:
    def __init__(self, X, x, L, d, s):
        self.X = X

        self.x = x
        self.L = L
        self.dc = len(L)
        self.d = d
        self.dg = d -self.dc
        self.s = s
        assert(type(X) == np.ndarray)
        dim = X.shape
        assert(len(dim) == 2)
        self.X_rows = dim[0]
        self.X_cols = dim[1]
        self.n = self.X_rows

        assert(type(x) == np.ndarray)
        dim = x.shape
        assert(len(dim) == 1)
        self.x_rows = dim[0]
        self.x_cols = 1
        self.b_rows = 1
        self.b_cols = 1
        self.b_size = self.b_rows * self.b_cols
        self.w_rows = self.X_cols
        self.w_cols = 1
        self.w_size = self.w_rows * self.w_cols

    def unpackParamVector(self, x):
        b = x[0]
        wAll = x[1:]
        
        v = np.insert(wAll[1:] , self.s, b)
        alpha_s = wAll[0] # linear parameter
        B_s = v[:self.dg] # s.row of covariance matrix
        
        i = self.dg # offset
        Rho_s= []
        for lr in self.L: # extract rho_{sr}
            j = i + lr
            Rho_s.append(v[i:j])
            j = i
            
        return (alpha_s, B_s, Rho_s)

    def getBounds(self):
        bounds = []
        bounds += [(1E-6, inf)] * self.b_size
        bounds += [(-inf, inf)] * (self.d - self.dc)
        for lr in self.L:
            bounds += [(0,0)]
            bounds += [(-inf, inf)] * (lr - 1)
        return bounds
        
    def getStartingPoint(self):
        return np.zeros(self.b_size + self.w_size)

    def functionValue(self, x):
        self.b = x[0 : 0 + self.b_size]
        self.w = x[0 + self.b_size : 0 + self.b_size + self.w_size]
        Xwdbx = np.add(self.x, np.divide(np.dot(self.X, self.w), self.b))
        f = self.b * np.dot(np.transpose(Xwdbx), Xwdbx) / self.n - np.log(self.b)
        return f
    def gradient(self, x):
        self.b = x[0 : 0 + self.b_size]
        self.w = x[0 + self.b_size : 0 + self.b_size + self.w_size]
        g = np.empty((1, 0 + self.b_size + self.w_size))
        g[0 : 1, 0 : 0 + self.b_size] = (np.add(np.add(np.add(np.multiply(np.dot(np.transpose(np.divide(np.dot(self.X, self.w), np.multiply(self.b, self.b))), np.add(self.x, np.divide(np.dot(self.X, self.w), self.b))), np.negative(np.divide(self.b, self.n))), np.divide(np.dot(np.transpose(np.add(self.x, np.divide(np.dot(self.X, self.w), self.b))), np.add(self.x, np.divide(np.dot(self.X, self.w), self.b))), self.n)), np.negative(np.dot(np.transpose(np.multiply(np.divide(self.b, self.n), np.add(self.x, np.divide(np.dot(self.X, self.w), self.b)))), np.divide(np.dot(self.X, self.w), np.multiply(self.b, self.b))))), np.divide(-1.0, self.b)))
        g[0 : 1, 0 + self.b_size : 0 + self.b_size + self.w_size] = (np.multiply(2.0, np.dot(np.transpose(np.divide(np.multiply(np.divide(self.b, self.n), np.add(self.x, np.divide(np.dot(self.X, self.w), self.b))), self.b)), self.X)))
        return g
        
    def functionValueAndGradient(self, x, verb = False):
        f = self.functionValue(x)
        g = self.gradient(x).reshape(-1)
#        g[0 : 1, 0 : 0 + self.b_size] =
#        (np.add(np.add(np.add(np.multiply(np.dot(np.transpose(
#        np.divide(np.dot(self.X, self.w), np.multiply(self.b, self.b))),
#Xwdbx),
#np.negative(np.divide(self.b, self.n))), np.divide(np.dot(
#np.transpose(Xwdbx),
#Xwdbx), self.n)),
#np.negative(np.dot(np.transpose(np.multiply(np.divide(self.b, self.n),
#                    Xwdbx)),
#np.divide(np.dot(self.X, self.w), np.multiply(self.b, self.b))))),
#np.divide(-1.0, self.b)))
        return (f, g)

############################################################################### CAT
class GMNLP_CAT_SNR: # separate nodewise regression with params as in p.101
    def __init__(self, V, B, L, d, r):
        self.V = V
        self.B = B
        self.L = L 
        self.c = [0] # cumulative # of levels
        for v in self.L:
            self.c.append(self.c[-1] + v)
            
        self.r = r
        self.dc = len(L) # #variables
        self.dg = d - self.dc

        assert(type(V) == np.ndarray)
        dim = V.shape
        assert(len(dim) == 2)
        self.V_rows = dim[0] # n
        self.V_cols = dim[1] # n_r
        
        assert(type(B) == np.ndarray)
        dim = B.shape
        assert(len(dim) == 2)
        self.B_rows = dim[0] #n
        self.B_cols = dim[1] # L_r
        
        self.th_rows = self.B_cols
        self.th_cols = self.V_cols
        self.th_size = self.th_rows * self.th_cols
    
    def unpackParamVector(self, x): # convert param vector to more convenient list of mats
        # theta = (Phi_r(.) ---, Phi_{rj}(., .) ---), see p.101
   
        u = x[0: self.L[self.r]]
        
        i = self.L[self.r] # offset for mats
        
        Phis = []; 
        for l in range(0, self.r): 
            b = self.L[self.r] # L_r
            a = self.L[l]
            Phis.append (x[i:i + a*b].reshape((a, b))) # Phi_{lr}
            i += a * b

        for l in range(self.r + 1, self.dc):
            a = self.L[self.r] # L_r
            b = self.L[l]
            Phis.append (x[i:i + a*b].reshape((a, b))) # Phi_{rl}
            i += a * b
        
        Rhos = []
        for s in range(self.dg):
            j = i + self.L[self.r]
            Rhos.append(x[i: j]) # rho_{sr}
            i = j
            
        return (u, Phis, Rhos)

    def _buildTheta(self, x):
        u, Phis, Rhos = self.unpackParamVector(x)
        
        Theta = np.zeros((self.L[self.r], 1 + self.c[self.dc] - self.L[self.r] + self.dg))
        Theta[:, 0] = u
        i = 1 # offset
        for l in range(self.r):
            j = i + self.L[l]
            Theta[:, i:j] = Phis[l].T #A, subtract L_r since this is not in mat
            i = j
            
        for l in range(self.r+1, self.dc):
            j = i + self.L[l]
            Theta[:, i:j] = Phis[l - 1]
            i = j

        #Rhos
        for l in range(self.dg):
            Theta[:, i] = Rhos[l]
            i += 1
         
        return Theta
        
    def packParamVector(self, dTheta): # and calc grad
        theta = np.zeros(0)
        
        u=dTheta[:, 0]
        i = 1 # offset
        
        Phis = []
        for l in range(self.r):
            j = i + self.L[l]
            Phis.append( (dTheta[:, i:j]).T)
            i = j
        for l in range(self.r+1, self.dc):
            j = i + self.L[l]
            Phis.append (dTheta[:,  i:j])
            i = j
        Rhos = []
        for l in range(self.dg):
            Rhos.append(dTheta[:, i])
            i += 1

        theta = np.append(theta, u)
        for j in range(self.dc - 1):
            theta = np.append(theta, Phis[j].ravel()) 
        for j in range(self.dg):
            theta = np.append(theta, Rhos[j].ravel()) 
       
        return theta # = grad
        
    def getBounds(self):
#        print(self.c)
        bounds = [(0, 0)]
        # set bounds to 0 for identifiability
        bounds += [(-inf, inf)] * (self.L[self.r] - 1) # theta_r1
        l=[]
        # Phis
        for j in range(self.r):
            bounds+=[(0,0)] * self.L[self.r]
            l= [(0,0)] + [(-inf, inf)] * (self.L[self.r] - 1)
            bounds += l * (self.L[j] - 1)
        for j in range(self.r + 1, self.dc): 
            bounds+=[(0,0)] * self.L[j]
            l= [(0,0)] + [(-inf, inf)] * (self.L[j] - 1)
            bounds += l * (self.L[self.r] - 1)
        # Rhos
        l= [(0,0)] + [(-inf, inf)] * (self.L[self.r] - 1)
        bounds += l * self.dg

        return bounds

    def getStartingPoint(self):
        return np.zeros(self.th_size) 

    def functionValueAndGradient(self, x, verb= False): # x = thetas as vector
        Theta = self._buildTheta(x)
        # uses gradients in rows        
    
        (n, nr) = self.V.shape
        #print (n, nr)
        (n, Lr) = self.B.shape
        #print(n, Lr)
        A = np.exp(np.dot(self.V, Theta.T))

        Atilde = np.divide(A, (np.repeat(np.sum(A, axis=1),Lr).reshape(n, Lr)))     
        f = -np.sum(np.log( np.sum(np.multiply(Atilde, self.B), axis=1)))

        dTheta = - np.dot(np.transpose(self.B - Atilde), self.V) #mat
##        print (grad)
        if verb: print('\n')
        if verb: print('f=', f)
         
        grad = self.packParamVector(dTheta)
        
        return (f, grad)  


      
