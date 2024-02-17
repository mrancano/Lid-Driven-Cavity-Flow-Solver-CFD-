import numpy as np
import scipy.linalg as sp
import os 
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import convectiveoperator,solutiontools,meshgen

N=3

utilsN = solutiontools.utilsClass(N) #real N
utilsN2 = solutiontools.utilsClass(N+2)
c = convectiveoperator.UconvectiveOperator(N+2)





u_vec = np.zeros((1,(N+1)*(N)))
i=0
for y in utilsN.y_vecdual:
    for x in utilsN.x_vec:
        u_vec[0,i] = x**2-y**2
        i+=1

print(utilsN2.incorpGhostBCsU(u_vec))

