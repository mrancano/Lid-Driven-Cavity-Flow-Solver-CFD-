import numpy as np
import scipy.linalg as sp
#generate v_hatxdiff
N=4

matrix = np.zeros((N, N))

# Set 1 on the diagonal above the main diagonal
np.fill_diagonal(matrix[:, :], 1)

    # Set -1 on the diagonal below the main diagonal
np.fill_diagonal(matrix[1:, :], -1)

#print(matrix)


#print(np.shape(np.kron(np.eye(N,dtype=int),matrix)))

x = np.arange(0,12,1)

#print(x.reshape((3,4))[1:-1,1:-1].flatten())
