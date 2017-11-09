# -*- coding: utf-8 -*-
"""
Copyright (c) 2017: Frank Nussbaum (frank.nussbaum@uni-jena.de), Philipp Lucas (philipp.lucas@uni-jena.de)
"""


import numpy as np
import pandas as pd


def cg_dummy():
    """Returns a dataframe that contains samples of a 4d cg distribution. See the code for the used parameters.
    @author: Philipp Lucas
    """

    # chose fixed parameters
    mu_M_Jena = [0, 0]
    mu_F_Jena = [1, 3]
    mu_M_Erfurt = [-10, 1]
    mu_F_Erfurt = [-5, -6]
    p_M_Jena = 0.35
    p_F_Jena = 0.25
    p_M_Erfurt = 0.1
    p_F_Erfurt = 0.3
    S = [[3, 0.5], [0.5, 1]]
    dims = ['sex', 'city', 'age', 'income']
    # and a sample size
    samplecnt = 1000

    # generate samples for each and arrange in dataframe
    df_cat = pd.concat([
        pd.DataFrame([["M", "Jena"]] * round(samplecnt * p_M_Jena), columns=['sex', 'city']),
        pd.DataFrame([["F", "Jena"]] * round(samplecnt * p_F_Jena), columns=['sex', 'city']),
        pd.DataFrame([["M", "Erfurt"]] * round(samplecnt * p_M_Erfurt), columns=['sex', 'city']),
        pd.DataFrame([["F", "Erfurt"]] * round(samplecnt * p_F_Erfurt), columns=['sex', 'city'])
    ])

    df_num = pd.concat([
        pd.DataFrame(np.random.multivariate_normal(mu_M_Jena, S, round(samplecnt * p_M_Jena)),
                     columns=['age', 'income']),
        pd.DataFrame(np.random.multivariate_normal(mu_F_Jena, S, round(samplecnt * p_F_Jena)),
                     columns=['age', 'income']),
        pd.DataFrame(np.random.multivariate_normal(mu_M_Erfurt, S, round(samplecnt * p_M_Erfurt)),
                     columns=['age', 'income']),
        pd.DataFrame(np.random.multivariate_normal(mu_F_Erfurt, S, round(samplecnt * p_F_Erfurt)),
                     columns=['age', 'income'])
    ])
    df = pd.concat([df_cat, df_num], axis=1)
    return df

def genCatData(n, levels, seed = 10):
    """ uniform/ independent draws according to the levels given in <levels>"""
    if seed >0 :
        np.random.seed(seed)
    d = len(levels.keys())

    X = np.zeros((n, d))
    
    for i in range(0,n):
        for j in range(0,d):
            X[i, j] = np.random.choice(levels[j])
            
    return X
    
def genCatDataJEx(n, levels = None, seed = 10):
    """ example from GM script """
    if seed >0:
        np.random.seed(seed)

    probs  = [1.0/12, 7.0/32, 1.0/12, 7.0/96, 1.0/4, 1.0/32, 1.0/4, 1.0/96]
    cprobs=[0]
    for p in probs:
        cprobs.append(cprobs[-1] + p)
        
    X = np.zeros((n, 3))
    
    for i in range(0,n):
        r = np.random.rand()
        if cprobs[0]<= r<= cprobs[1]: a =[1,1,1]
        if cprobs[1]<= r<= cprobs[2]: a =[1,1,0]
        if cprobs[2]<= r<= cprobs[3]: a =[1,0,1]
        if cprobs[3]<= r<= cprobs[4]: a =[1,0,0]
        if cprobs[4]<= r<= cprobs[5]: a =[0,1,1]
        if cprobs[5]<= r<= cprobs[6]: a =[0,1,0]
        if cprobs[6]<= r<= cprobs[7]: a =[0,0,1]
        if cprobs[7]<= r<= cprobs[8]: a =[0,0,0]
        X[i, :] = a
    return X

def genCGSample(n, d):
    """ pass a dictionary d with options fun, Sigma, catvalasmean, levels, seed"""
    fun = d['fun']
    catvalasmean = d['catvalasmean']
    levels = d['levels']
    dc = len(levels.keys())
    seed = d['seed']

    Sigma = d['Sigma']
    dg = Sigma.shape[0]
    
    if catvalasmean and dg!=dc:
        print('SamplingWarning: dg!=dc, using average of categoricals for each mean component')
    
    np.random.seed(seed)
    
    data = np.empty((n, dc+dg))
    
    for i in range(n):
        x = fun(1, levels = levels, seed = -1)
        # sample gaussian variables according to p_{\mu(x), \Sigma{X}}
        k = 10.0
        if catvalasmean:
            if dg == dc:
                j = 4 * x[0][0] + 2 * x[0][1] + x[0][2]
                j = sum(x[0])
                j = j*k

                mu = x.ravel() * k
                mu = [j] * dg
            else:
                mu =[np.sum(x)*k] * dg
            
        else:
            mu = [0,-5,10]
            mu = [0,0,0] # mu and Sigma independent of x!
        
        y = np.random.multivariate_normal(mu, Sigma, 1)

        data[i][0:dc] = x
        data[i][dc:dc +dg] = y
        
    catcols = ['c%d'%(i) for i in range(dc)]
    gausscols = ['g%d'%(i) for i in range(dg)]
    cols = catcols + gausscols
    df = pd.DataFrame(data=data, index=None, columns=cols) 

    df[catcols] = df[catcols].astype('object') # replace 0 bei 'no', and 1 bei 'yes' ???
    for c in cols[:dc]:
        for j in range(11): # have only values in range 10
            df.loc[df[c]==j, c] = 'val%d'%(j)
#        df.loc[df[c] ==0, c] = 'no'

    return df #data

def genMixGSample(n, dic): # outdated
    """ pass a dictionary d with options fun, Sigma, catvalasmean, levels, seed"""
#    fun = dic['fun']
#    catvalasmean = dic['catvalasmean']
    levels = dic['levels']
    dc = len(levels.keys())
    latents = dic['latents']
#    dl = len(latents.keys())
    seed = dic['seed']

    Sigma = dic['Sigma']
    dg = Sigma.shape[0]


    np.random.seed(seed)
    
    data = np.empty((n, dc+dg))
    
    for i in range(n):
        z = genCatData(1, levels = latents, seed = -1) # latent variable -> which gaussian in the mixture?
#        x = fun(1, levels = levels, seed = -1)
        # sample gaussian variables according to p_{\mu(x), \Sigma{X}}

        mu = [z[0,0]*5] *dg

        
        y = np.random.multivariate_normal(mu, Sigma, 1)

#        data[i][0:dc] = x
        data[i][dc:dc +dg] = y
    return data