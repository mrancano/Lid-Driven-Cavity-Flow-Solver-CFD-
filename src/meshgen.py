import numpy as np
#consider adding plotting function
class meshGenClass:
    def __init__(self,N): #i: 0,1,2,...,N
                          #j: 0,1,2,...,N
        self.N = N
        self.h = 1/self.N
        self.i = np.arange(0,self.N+1,1) #to include last point as N
        self.j = np.arange(0,self.N+1,1) #to include last point as N
        self.x_vec = self.h*self.i
        self.y_vec = self.h*self.j
        self.x_vecdual = (self.x_vec[:self.N]+self.x_vec[1:])*0.5
        self.y_vecdual = (self.y_vec[:self.N]+self.y_vec[1:])*0.5
        self.x_mesh , self.y_mesh = np.meshgrid(self.x_vec,self.y_vec,sparse=False) #for plotting 3d needs to be false for
                                                                                    #for evaluating functions can be true
        self.x_meshdual,self.y_meshdual = np.meshgrid(self.x_vecdual,self.y_vecdual,sparse=False)
        self.x_meshU ,self.y_meshU = np.meshgrid(self.x_vec,self.y_vecdual,sparse=False)
        self.x_meshV ,self.y_meshV = np.meshgrid(self.x_vecdual,self.y_vec,sparse=False)

    def genXYvec(self):
        x_vec = self.h*self.i
        y_vec = self.h*self.j
        return x_vec,y_vec
    
    def genXYmesh(self):
        x_mesh,y_mesh = np.meshgrid(self.x_vec,self.y_vec,sparse=False)
        return x_mesh,y_mesh
    
    def genXYvecdual(self):
        x_vecdual = (self.x_vec[:self.N]+self.x_vec[1:])*0.5
        y_vecdual = (self.y_vec[:self.N]+self.y_vec[1:])*0.5

        return x_vecdual,y_vecdual

    def genXYmeshdual(self): #for phi
        x_meshdual,y_meshdual = np.meshgrid(self.x_vecdual,self.y_vecdual,sparse=False)
        return x_meshdual,y_meshdual
    
    def genUmesh(self):
        x_meshU ,y_meshU = np.meshgrid(self.x_vec,self.y_vecdual,sparse=False)
        return x_meshU ,y_meshU
    
    def genVmesh(self):
        x_meshV ,y_meshV = np.meshgrid(self.x_vecdual,self.y_vec,sparse=False)
        return x_meshV ,y_meshV

    def unpackUvec(self,Uvec): #for unpacking vectors that lie on the Umesh (x,y_hat)
        Umatrix = np.reshape(Uvec,(self.N,self.N+1))
        return Umatrix #KEEP IN MIND MATRICES FROM THIS FUNCTION ARE NOT INDEXED [i,j] for that you have to transpose

    def unpackVvec(self,Vvec): #for unpacking vectors that lie on the Vmesh (x,y_hat)
        Vmatrix = np.reshape(Vvec,(self.N+1,self.N))
        return Vmatrix #KEEP IN MIND MATRICES FROM THIS FUNCTION ARE NOT INDEXED [i,j] for that you have to transpose
    
    def unpackphivec(self,phivec): #for unpacking vectors that lie on the phimesh (x,y_hat)
        phimatrix = np.reshape(phivec,(self.N,self.N))
        return phimatrix #KEEP IN MIND MATRICES FROM THIS FUNCTION ARE NOT INDEXED [i,j] for that you have to transpose
    
    
    



