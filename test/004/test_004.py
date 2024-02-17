import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import solutiontools
from src import meshgen

N = 3

utils = solutiontools.utilsClass(N)
meshGen = meshgen.meshGenClass(N)
##test uinterp
u_vec = np.zeros((1,(N+1)*(N)))

i=0
for y in utils.y_vecdual:
    for x in utils.x_vec:
        u_vec[0,i] = x**2-y**2
        i+=1

u_hat = utils.uinterp(u_vec)

#utils.plotgivenmesh(meshGen.x_meshdual,meshGen.y_meshdual,meshGen.unpackphivec(u_hat))

##test vinterp
v_vec = np.zeros((1,(N+1)*(N)))

i=0
for y in utils.y_vec:
    for x in utils.x_vecdual:
        v_vec[0,i] = x**2-y**2
        i+=1

v_hat = utils.vinterp(v_vec)

#utils.plotgivenmesh(meshGen.x_meshdual,meshGen.y_meshdual,meshGen.unpackphivec(v_hat))

print(utils.vinterp(np.arange(0,N*(N+1),1)))

print(np.arange(0,N*(N+1),1))
