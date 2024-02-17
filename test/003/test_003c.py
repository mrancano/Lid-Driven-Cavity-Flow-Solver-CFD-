#test of full convective operator (in U) with respect to an analytic solution

import numpy as np
import scipy.linalg as sp
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import convectiveoperator,solutiontools,meshgen
N=3
mesh = meshgen.meshGenClass(N)
utils = solutiontools.utilsClass(N)
Uconv = convectiveoperator.UconvectiveOperator(N)



u_vec = np.zeros((1,(N+1)*(N)))
v_vec = np.zeros((1,(N+1)*(N)))

i=0
for y in utils.y_vecdual:
    for x in utils.x_vec:
        u_vec[0,i] = x+y
        i+=1
i=0
for y in utils.y_vec:
    for x in utils.x_vecdual:
        v_vec[0,i] = x-y
        i+=1

#exact solution 
exact = -((mesh.x_meshU+mesh.y_meshU) + (mesh.x_meshU-mesh.y_meshU))[1:-1,1:-1]
#exact = ((mesh.x_meshU**2-mesh.y_meshU**2)*2*mesh.y_meshU)[1:-1,1:-1]

#utils.plotgivenmesh(mesh.x_meshU[1:-1,1:-1],mesh.y_meshU[1:-1,1:-1],exact[1:-1,1:-1])

#using convective operator
conv_vec = Uconv.apply(u_vec,utils.vinterp(v_vec))

numeric = conv_vec.reshape(((N-2),(N-1)))
print(np.mean((exact-numeric)**2))

