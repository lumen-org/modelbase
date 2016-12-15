# -*- coding: utf-8 -*-
"""
Copyright: Frank Nussbaum (frank.nussbaum@uni-jena.de)
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



def plothist(x): # https://plot.ly/matplotlib/histograms/
#    print(len(x))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # the histogram of the data
    n, bins, patches = ax.hist(x, 50, histtype='bar', facecolor='green', alpha=0.75, stacked = True) # normed=1, 
    
    # hist uses np.histogram under the hood to create 'n' and 'bins'.
    # np.histogram returns the bin edges, so there will be 50 probability
    # density values in n, 51 bin edges in bins and 50 patches.  To get
    # everything lined up, we'll compute the bin centers
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
#    y = mlab.normpdf( bincenters, mu, sigma)
#    print(bincenters)
#    print(n)
#    l = ax.plot(bincenters,  'r--', linewidth=1)
    
    ax.set_xlabel('Smarts')
    ax.set_ylabel('Rel. Freq.')
    #ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#    ax.set_xlim(-5, 5)
#    ax.set_ylim(0, 0.2)
    ax.grid(True)
    
    plt.show()