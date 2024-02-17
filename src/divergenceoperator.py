import numpy as np

class DivergenceOp():
    def __init__(self,N,h):
        self.N = N
        self.h = h
        #generate matrix for u_i+1,j-u_i,j
        matrix = np.zeros((self.N+1, self.N+1))
        #1 on the diagonal above the main diagonal
        np.fill_diagonal(matrix[:, 1:], 1)
        # Set -1 on the diagonal below the main diagonal
        np.fill_diagonal(matrix[:, :], -1) 
        #block diagonal matrix with N blocks
        self.Uxdiv = np.kron(np.eye(self.N,dtype=int),matrix)

        #generate matrix for v_i,j+1-v_i,j
        #generate Vydiff
        matrix = np.zeros(((self.N+1)*self.N,(self.N+1)*self.N))
        # Set 1 on the j+1 diagonal
        np.fill_diagonal(matrix[:, (self.N):], 1)
        # Set -1 on the main diagonal
        np.fill_diagonal(matrix[:, :], -1)
        #block diagonal matrix with N blocks
        self.Vydiv = matrix
        #print(self.Uxdiv)
        #print(self.Vydiv)

    def interiorizeUdiv(self,u_vec):
        uxdiv = np.matmul(self.Uxdiv,u_vec.T)
        uxdiv = uxdiv.reshape((self.N,self.N+1))[:,:-1].flatten() #need to test (i got rid of only the last column ) N=3 output is 3 by 3

        return uxdiv
    
    def interiorizeVdiv(self,v_vec):
        vydiv = np.matmul(self.Vydiv,v_vec.T)
        vydiv = vydiv.reshape((self.N+1,self.N))[:-1,:].flatten() #need to test (i got rid of only the last row ) N=3 output is 3 by 3

        return vydiv
    
    def apply(self,u_vec,v_vec):
        return (self.interiorizeUdiv(u_vec)+self.interiorizeVdiv(v_vec))/self.h

