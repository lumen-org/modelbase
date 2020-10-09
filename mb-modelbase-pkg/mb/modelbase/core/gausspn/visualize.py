# -*- coding: utf-8 -*-
"""
@author: Julien Klaus
@email: julien.klaus@uni-jena.de
"""

from graphviz import Digraph
import numpy as np

index = 0


def generateSPNPdf (spn,filename="img/gausspn") :
    """
    Generates a PDF with a visualisation of an SPN
    """
    
    dot = Digraph(format="pdf")
    
    worklist = []
    used = []
    
    root = spn.root.children[0]
    worklist.append((index,root))
    # id of each node is unique
    used.append(index)
   
    while len(worklist) > 0:
        (ind,node) = worklist.pop(0)
        #add the current node
        if type(node).__name__ not in ["SumNode","ProductNode"]:
            #TODO: Label für Verteilungen der Blätter
            dot.node(str(ind),shape="record",label="Variable: "+str(node.index)+"\\nMean: "+str(node.mean)+"\\nVar: "+str(node.var))
        else:
            dot.node(str(ind), label="+" if type(node).__name__ is "SumNode" else \
                     "*" if type(node).__name__ is "ProductNode" else str(ind))
        #there are children
        if type(node).__name__ is "SumNode" or type(node).__name__ is "ProductNode":
            for i,child in zip(range(len(node.children)),node.children):
               #draw a line
               cIndex = len(used)+1
               while cIndex in used:
                  cIndex += 1
               if type(node).__name__ is "SumNode":
                  dot.edge(str(ind),str(cIndex),label=str(np.exp(node.get_log_weights())[i]))
               else:
                  dot.edge(str(ind),str(cIndex))
               #do it next time
               worklist.append((cIndex,child))
               used.append(cIndex)
    dot.render(filename, cleanup=True)