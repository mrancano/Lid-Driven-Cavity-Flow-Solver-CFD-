import numpy as np
import scipy.linalg as sp
import os 
import sys
import matplotlib.pyplot as plt

class phigradientX():
    def __init__(self,N,h):
        self.N = N
        self.h = h
        #generate matrix for phi_i,j-phi_i-1,j
        matrix = np.zeros((self.N, self.N))
        #1 on the main diagonal
        np.fill_diagonal(matrix[:, :], 1)
        # Set -1 on the diagonal below the main diagonal
        np.fill_diagonal(matrix[1:, :], -1) 
        #block diagonal matrix with N blocks
        self.phigrad = np.kron(np.eye(self.N,dtype=int),matrix)

        #print(self.phigrad)
        

    def applyandinteriorize(self,phi): 
        #I AM OUTPUTTING THE APPLY WITH ONLY THE INTERIOR POINTS EVEN THO
        #MORE POINTS ARE VALID FOR THE ACTUAL MATRIX. PHIGRAD WHICH CAN BE USED IMPLICITLY
        #HAS N*N,N*N SO THAT IT CAN BE MULTIPLIED BY A PHI WHICH INCLUDES GHOST POINTS
        phigrad = np.matmul(self.phigrad,phi.T)

        

        phigrad = phigrad.reshape((self.N,self.N))[1:-1,1:].flatten() #only get rid of leftmost column and top and bottom rows

        #print(phigrad.reshape(self.N-2,self.N-1))

        return phigrad/self.h

class phigradientY():
    def __init__(self,N,h):
        self.N = N
        self.h = h
        #generate matrix for phi_i,j-phi_i,j-1
        matrix = np.zeros((self.N*self.N, self.N*self.N))
        #1 on the the main diagonal
        np.fill_diagonal(matrix[:, :], 1)
        # Set -1 on the diagonal N below the main
        np.fill_diagonal(matrix[self.N:, :], -1) 
        
        self.phigrad = matrix

        
        

    def applyandinteriorize(self,phi): 
        #I AM OUTPUTTING THE APPLY WITH ONLY THE INTERIOR POINTS EVEN THO
        #MORE POINTS ARE VALID FOR THE ACTUAL MATRIX. PHIGRAD WHICH CAN BE USED IMPLICITLY
        #HAS N*N,N*N SO THAT IT CAN BE MULTIPLIED BY A PHI WHICH INCLUDES GHOST POINTS
        phigrad = np.matmul(self.phigrad,phi.T)

        phigrad = phigrad.reshape((self.N,self.N))[1:,1:-1].flatten() #only get rid of bottom row

        #print(phigrad.reshape(self.N-1,self.N-2))

        return phigrad/self.h


        



