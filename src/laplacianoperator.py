import numpy as np
import scipy.linalg as sp

class Ulaplacian():
    def __init__(self,N,h): #so G ghost points can be input while maintaining accurate h
        self.N = N
        self.h = h

        #generate matrix for four point laplacian approx
        matrix = np.zeros((self.N+1, self.N+1))
        #1 on the diagonal above the main diagonal u_i+1
        np.fill_diagonal(matrix[:, 1:], 1)
        #-4 on main diagonal uij
        np.fill_diagonal(matrix[:, :], -4)
        # Set 1 on the diagonal below the main diagonal u_i-1
        np.fill_diagonal(matrix[1:, :], 1) 
        matrix = np.kron(np.eye(self.N,dtype=int),matrix)
        # Set 1 on the j+1 diagonal
        np.fill_diagonal(matrix[:, (self.N+1):], 1)
        # Set 1 on the diagonal j-1 diagonal
        np.fill_diagonal(matrix[(self.N+1):, :], 1)
        #block diagonal matrix with N blocks
        self.ULap= matrix/self.h**2
    
    def interiorizeUlap(self,u_vec):
        u_lap = np.matmul(self.ULap,u_vec.T)
        u_lap = u_lap.reshape((self.N,self.N+1))[1:-1,1:-1].flatten()

        return u_lap
    
    def applyUlap(self,u_vec):
        return self.interiorizeUlap(u_vec)

class Vlaplacian():
    def __init__(self,N,h):
        self.N = N
        self.h = h


        #generate matrix for four point laplacian approx
        matrix = np.zeros((self.N, self.N))
        #1 on the diagonal above the main diagonal v_i+1
        np.fill_diagonal(matrix[:, 1:], 1)
        #-4 on main diagonal vij
        np.fill_diagonal(matrix[:, :], -4)
        # Set 1 on the diagonal below the main diagonal v_i-1
        np.fill_diagonal(matrix[1:, :], 1) 
        matrix = np.kron(np.eye(self.N+1,dtype=int),matrix)
        # Set 1 on the j+1 diagonal
        np.fill_diagonal(matrix[:, (self.N):], 1)
        # Set 1 on the diagonal j-1 diagonal
        np.fill_diagonal(matrix[(self.N):, :], 1)
        #block diagonal matrix with N blocks
        self.VLap= matrix/self.h**2

    def interiorizeVlap(self,v_vec):
        v_lap = np.matmul(self.VLap,v_vec.T)
        v_lap = v_lap.reshape((self.N+1,self.N))[1:-1,1:-1].flatten() #this shape might matter but on other tests it doesnt
                                                                    # in the sense that this or the transpose leads is same
        return(v_lap)
    
    def applyVlap(self,v_vec):
        return self.interiorizeVlap(v_vec)
    
class Philaplacian():
    def __init__(self,N,h):
        self.N = N
        self.h = h

        #generate matrix for four point laplacian approx
        matrix = np.zeros((self.N, self.N))
        #1 on the diagonal above the main diagonal v_i+1
        np.fill_diagonal(matrix[:, 1:], 1)
        #-4 on main diagonal vij
        np.fill_diagonal(matrix[:, :], -4)
        # Set 1 on the diagonal below the main diagonal v_i-1
        np.fill_diagonal(matrix[1:, :], 1) 
        matrix = np.kron(np.eye(self.N,dtype=int),matrix)
        # Set 1 on the j+1 diagonal
        np.fill_diagonal(matrix[:, (self.N):], 1)
        # Set 1 on the diagonal j-1 diagonal
        np.fill_diagonal(matrix[(self.N):, :], 1)
        #block diagonal matrix with N blocks
        self.PhiLap = matrix/self.h**2

    def interiorizePhilap(self,v_vec):
        phi_lap = np.matmul(self.PhiLap,v_vec.T)
        phi_lap = phi_lap.reshape((self.N,self.N))[1:-1,1:-1].flatten() 
        return(phi_lap)
    
    def applyPhilap(self,v_vec):
        return self.interiorizePhilap(v_vec)

