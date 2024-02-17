import numpy as np
import scipy.linalg as sp
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import convectiveoperator,solutiontools

N=3
e = convectiveoperator.UconvectiveOperator(N)
utils = solutiontools.utilsClass(N)

#print(e.Uxdiff)
#print(e.Vhatxdiff)

#print(e.Uydiff[5:15])

#print(np.shape(e.Uxdiff))
#print(np.shape(e.Uydiff))

u_vec = np.zeros((1,(N+1)*(N)))
v_vec = np.zeros((1,(N+1)*(N)))

i=0
for y in utils.y_vecdual:
    for x in utils.x_vec:
        u_vec[0,i] = x**2-y**2
        i+=1
i=0
for y in utils.y_vec:
    for x in utils.x_vecdual:
        v_vec[0,i] = x**2-y**2
        i+=1
v_hat = utils.vinterp(v_vec)

#print(np.matmul(e.Vhatxdiff,v_hat.T))

#print(np.matmul(e.Vhatxdiff,v_hat.T).reshape((N,N))[1:-1,1:].flatten())


#generate uxdiff


matrix = np.zeros((N+1, N+1))

# Set 1 on the diagonal above the main diagonal
np.fill_diagonal(matrix[:, 1:], 1)

    # Set -1 on the diagonal below the main diagonal
np.fill_diagonal(matrix[1:, :], -1)

#print(matrix)


#print(np.kron(np.eye(N,dtype=int),matrix))

#print(np.matmul(e.Vhatxdiff,np.arange(0,N*N)**2))

#print(e.interiorizeVdiffs(((np.arange(0,N*N,1))**2).T))

#print(e.interiorizeUdiffs(((np.arange(0,N*(N+1),1))**2).T)[1])

#e.apply(((np.arange(0,N*(N+1),1))**2).T,((np.arange(0,N*(N),1))**2).T)

print(e.Vhatxdiff)
