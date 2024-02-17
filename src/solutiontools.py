import numpy as np
import matplotlib.pyplot as plt

class utilsClass():
    def __init__(self,N):
        self.N = N
        self.G = N+2
        self.h = 1/self.N
        self.i = np.arange(0,self.N+1,1) #to include last point as N
        self.j = np.arange(0,self.N+1,1) #to include last point as N
        self.x_vec = self.h*self.i
        self.y_vec = self.h*self.j
        self.x_vecdual = (self.x_vec[:self.N]+self.x_vec[1:])*0.5
        self.y_vecdual = (self.y_vec[:self.N]+self.y_vec[1:])*0.5

    def plotgivenmesh(self,meshx,meshy,unpackedvec):
        hdual = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(meshx,meshy,unpackedvec,50)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def uinterp(self,u_vec):
        u_matrix = np.reshape(u_vec,(self.N,self.N+1))
        u_interpmatrix = (u_matrix[:,1:]+u_matrix[:,:self.N])*0.5
        u_hat = np.reshape(u_interpmatrix,(1,self.N*self.N))

        return u_hat
    
    def vinterp(self,v_vec):
        v_matrix = np.reshape(v_vec,(self.N+1,self.N))
        
        v_interpmatrix = (v_matrix[1:,:]+v_matrix[:self.N,:])*0.5
        
        v_hat = np.reshape(v_interpmatrix,(1,self.N*self.N))

        return v_hat

    #def stripxboundary() #consider adding function for stripping the x boundary points of a vector

    #for testing NOT IMPLEMENTED YET MAY NOT BE NEEDED
    def incorpGhostBCsU(self,u_vec): #assume N+2 is passed into function and u_vec is for domain of N
        u_fullmat = np.zeros((self.N,self.N+1)) #for N = 3 the input is 5, valid domain is 3 by 4, and including ghost is 5 by 6
        u_fullmat[1:-1,1:-1]= np.reshape(u_vec,((self.N-2,self.N-1))) #for input 5 valid domain is 3 by 4

        #apply bottom boundary for ghost points knowing that u is 0 on boundary so we take u_ghost as - the point above
        u_fullmat[0,1:-1] = -u_fullmat[1,1:-1]
        #apply left boundary for ghost points knowing that u is 0 on boundary so we take u_ghost as - the point above
        u_fullmat[0,1:-1] = -u_fullmat[1,1:-1]


        return u_fullmat
    
    def genSol(self):
        u_vec = np.zeros((1,(self.N+1)*(self.N)))
        v_vec = np.zeros((1,(self.N+1)*(self.N)))

        i=0
        for y in self.y_vecdual:
            for x in self.x_vec:
                u_vec[0,i] = x**2-y**2
                i+=1
        i=0
        for y in self.y_vec:
            for x in self.x_vecdual:
                v_vec[0,i] = x**2+y**2
                i+=1

        return u_vec,v_vec
    
    def genPhi(self):
        phi_vec = np.zeros((1,(self.N)*(self.N)))

        i=0
        for y in self.y_vecdual:
            for x in self.x_vecdual:
                phi_vec[0,i] = x**2-y**2
                i+=1

        return phi_vec
    
    def genPearSol(self,t):
        u_vec = np.zeros((1,(self.N+1)*(self.N)))
        v_vec = np.zeros((1,(self.N+1)*(self.N)))

        i=0
        for y in self.y_vecdual:
            for x in self.x_vec:
                u_vec[0,i] = -np.cos(x*np.pi)*np.sin(y*np.pi)*np.exp(-2*t)
                i+=1
        i=0
        for y in self.y_vec:
            for x in self.x_vecdual:
                v_vec[0,i] = np.sin(x*np.pi)*np.cos(y*np.pi)*np.exp(-2*t)
                i+=1

        return u_vec,v_vec
    
    def genPearPhi(self,t):
        phi_vec = np.zeros((1,(self.N)*(self.N)))

        i=0
        for y in self.y_vecdual:
            for x in self.x_vecdual:
                phi_vec[0,i] = -1/4*(np.cos(2*x*np.pi)+np.cos(2*y*np.pi))*np.exp(-4*t)
                i+=1

        return phi_vec
    
    def applyUBCs(self,u_vec): #input is N*N+1 vector output is G*G+1
        u_vec = u_vec.reshape((self.N,self.N+1))

        output = np.zeros((self.G,self.G+1))

        output[1:-1,1:-1] = u_vec

        #bottom boundary (second through second to last point in bottom row)
        output[0,1:self.G] = -u_vec[0,:self.N+1]

        #top boundary (second through second to last point in top row)
        output[-1,2:self.G-1] = 2-u_vec[-1,1:self.N]

        return output.flatten()

    def applyVBCs(self,v_vec):
        v_vec = v_vec.reshape((self.N+1,self.N))

        output = np.zeros((self.G+1,self.G))

        output[1:-1,1:-1] = v_vec

        #left boundary (second through second to last point in left column)
        output[1:self.G,0] = -v_vec[:self.N+1,0]

        #right boundary (second through second to last point in right column)
        output[1:self.G,-1] = -v_vec[:self.N+1,-1]

        #print(output)

        return output.flatten()
    
    def applyPHIBCs(self,phi_vec):
        phi_vec = phi_vec.reshape((self.N,self.N))

        output = np.zeros((self.G,self.G))

        output[1:-1,1:-1] = phi_vec

        #left boundary (second through second to last point in left column)
        output[1:self.G-1,0] = phi_vec[:,0]

        #right boundary (second through second to last point in right column)
        output[1:self.G-1,-1] = phi_vec[:,-1]

        #bottom boundary (second through second to last point in bottom row)
        output[0,1:self.G-1] = phi_vec[0,:]

        #top boundary (second through second to last point in top row)
        output[-1,1:self.G-1] = phi_vec[-1,:]

        #print(output)

        return output.flatten()
