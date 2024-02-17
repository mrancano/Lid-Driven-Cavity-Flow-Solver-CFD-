import numpy as np
import scipy.linalg as sp

class UconvectiveOperator():
    def __init__(self,N,h):
        self.N = N #including ghost
        self.h = h
        #generate matrix for u_i+1,j-u_i-1,j
        matrix = np.zeros((self.N+1, self.N+1))
        #1 on the diagonal above the main diagonal
        np.fill_diagonal(matrix[:, 1:], 1)
        # Set -1 on the diagonal below the main diagonal
        np.fill_diagonal(matrix[1:, :], -1) 
        #block diagonal matrix with N blocks
        self.Uxdiff = np.kron(np.eye(self.N,dtype=int),matrix)

        #generate v_hatxdiff #NOT DIFF
        matrix = np.zeros((self.N,self.N))
        # Set 1 on the main diagonal
        np.fill_diagonal(matrix[:, :], 1)
        # Set -1 on the diagonal below the main diagonal
        np.fill_diagonal(matrix[1:, :], 1)
        #block diagonal matrix with N blocks
        self.Vhatxdiff = np.kron(np.eye(self.N,dtype=int),matrix)

        #generate uydiff
        matrix = np.zeros(((self.N+1)*self.N,(self.N+1)*self.N))
        # Set 1 on the j+1 diagonal
        np.fill_diagonal(matrix[:, (self.N+1):], 1)
        # Set -1 on the diagonal j-1 diagonal
        np.fill_diagonal(matrix[(self.N+1):, :], -1)
        #block diagonal matrix with N blocks
        self.Uydiff = matrix
    
    def interiorizeUdiffs(self,u_vec): #applies matrices and filters valid points
        uxdiff = np.matmul(self.Uxdiff,u_vec.T)
        uxdiff = uxdiff.reshape((self.N,self.N+1))[1:-1,1:-1].flatten()
        uydiff = np.matmul(self.Uydiff,u_vec.T)
        uydiff = uydiff.reshape((self.N,self.N+1))[1:-1,1:-1].flatten() #tested twice
        u_vec = u_vec.reshape((self.N,self.N+1))[1:-1,1:-1].flatten()

        

        return uxdiff,uydiff,u_vec
    
    def interiorizeVdiffs(self,vhat_vec): #applies matrices and filters valid points #NOT DIFFF
        vhatxdiff = np.matmul(self.Vhatxdiff,vhat_vec.T)
        #print(vhat_vec)

        
        
        vhatxdiff = vhatxdiff.reshape((self.N,self.N))[1:-1,1:].flatten() #tested twice
        
        return vhatxdiff
    

        

    def apply(self,u_vec,v_hatvec):
        uxdiff,uydiff,u_valid = self.interiorizeUdiffs(u_vec)
        vhatxdiff = self.interiorizeVdiffs(v_hatvec)

        #print(vhatxdiff)
        #print(uydiff)
    

        #print(-0.5*np.multiply(vhatxdiff,uydiff))
        #print(-np.multiply(u_valid,uxdiff))
        Uconvective = -np.multiply(u_valid,uxdiff)-0.5*np.multiply(vhatxdiff,uydiff) #tested twice last term
        
        return Uconvective/(2*self.h)
    
class VconvectiveOperator():
    def __init__(self,N,h):#both h and N input so that ghost point N values can be input without changing h
        self.N = N
        self.h = h
        #generate matrix for v_i+1,j-v_i-1,j
        matrix = np.zeros((self.N, self.N))
        #1 on the diagonal above the main diagonal
        np.fill_diagonal(matrix[:, 1:], 1)
        # Set -1 on the diagonal below the main diagonal
        np.fill_diagonal(matrix[1:, :], -1) 
        #block diagonal matrix with N+1 blocks
        self.Vxdiff = np.kron(np.eye(self.N+1,dtype=int),matrix)

        #generate u_hatyinterp
        matrix = np.zeros((self.N*self.N,self.N*self.N))
        # Set 1 on the main diagonal
        np.fill_diagonal(matrix[:, :], 1)
        # Set 1 on the diagonal j-1 diagonal
        np.fill_diagonal(matrix[(self.N):, :], 1)
        #block diagonal matrix with N blocks
        self.Uyinterp = matrix

        #generate Vydiff
        matrix = np.zeros(((self.N+1)*self.N,(self.N+1)*self.N))
        # Set 1 on the j+1 diagonal
        np.fill_diagonal(matrix[:, (self.N):], 1)
        # Set -1 on the diagonal j-1 diagonal
        np.fill_diagonal(matrix[(self.N):, :], -1)
        #block diagonal matrix with N blocks
        self.Vydiff = matrix
    
    def interiorizeVdiffs(self,v_vec): #applies matrices and filters valid points
        vxdiff = np.matmul(self.Vxdiff,v_vec.T)
        vxdiff = vxdiff.reshape((self.N+1,self.N))[1:-1,1:-1].flatten()
        vydiff = np.matmul(self.Vydiff,v_vec.T)
        vydiff = vydiff.reshape((self.N,self.N+1))[1:-1,1:-1].flatten() #make sure this shape is coherent in future (right now it doesnt matter because we take rows and columns out symmetrically)
        v_vec = v_vec.reshape((self.N+1,self.N))[1:-1,1:-1].flatten()

        return vxdiff,vydiff,v_vec
    
    def interiorizeUdiffs(self,uhat_vec): #applies matrices and filters valid points #NOT DIFFF
        
        uhatydiff = np.matmul(self.Uyinterp,uhat_vec.T)
        
        uhatydiff = uhatydiff.reshape((self.N,self.N))[1:,1:-1].flatten() 

        return uhatydiff
        

    def apply(self,u_hatvec,v_vec):
        vxdiff,vydiff,v_valid = self.interiorizeVdiffs(v_vec)
        uhatxdiff = self.interiorizeUdiffs(u_hatvec)
    

        #print(-0.5*np.multiply(vhatxdiff,uydiff))
        #print(-np.multiply(u_valid,uxdiff))
        Vconvective = -0.5*np.multiply(uhatxdiff,vxdiff)-np.multiply(v_valid,vydiff) 
        #print(Vconvective)
        
        return Vconvective/(2*self.h)
    

        
