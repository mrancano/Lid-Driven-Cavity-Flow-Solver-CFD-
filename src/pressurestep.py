import numpy as np
import laplacianoperator,divergenceoperator,gradientoperator
import scipy as sp

class pressureCalc(): 
    def __init__(self,N,k):
        G = N+2
        self.h = 1/N
        L = laplacianoperator.Philaplacian(G,self.h)
        A_pre = L.PhiLap
# BOUNDARY CONDITIONS ARE SUCH THAT THE GRADIENT OF PRESSURE IS 0 AT BOUNDARIES (SO THE GHOST POINTS ARE EQUAL TO THE INTERIOR POINTS)
        #bottom row
        bottomatrix = np.eye(G*G)
        np.fill_diagonal(bottomatrix[:,G:],-1)
        A_pre[:G] = bottomatrix[:G]
        #top row
        topmatrix = np.eye(G*G)
        np.fill_diagonal(topmatrix[G:,:],-1)
        A_pre[-G:] = topmatrix[-G:]
        #left column
        leftmatrix = np.eye(G*G)
        np.fill_diagonal(leftmatrix[:,1:],-1)
        A_pre[::G] = leftmatrix[::G]
        #right column
        leftmatrix = np.eye(G*G)
        np.fill_diagonal(leftmatrix[1:,:],-1)
        A_pre[(G-1)::G] = leftmatrix[(G-1)::G]

        #set useless points to identity (Corner points)
        im = np.eye(G*G)
        A_pre[0] = im[0]
        A_pre[G-1] = im[G-1]
        A_pre[-G] = im[-G]
        A_pre[-1] = im[-1]

        #print(A_pre)

        self.A = k*A_pre

        self.A_wlagrangian = np.zeros((G*G+1,G*G+1))
        self.A_wlagrangian[:-1,:-1] = self.A
        self.A_wlagrangian[:-1,-1] = np.ones((G*G,1))[0]
        self.A_wlagrangian[-1,:-1] = np.ones((1,G*G))[0]

        #print(self.A_wlagrangian)
    
    def generateb(self,N,ustar,vstar):
        G = N+2
        D = divergenceoperator.DivergenceOp(G,self.h)
        rhs1 = D.apply(ustar,vstar) #Output should be of G*G length
        #print(rhs1)
        #bottom row
        rhs1[:G] = 0
        #top row
        rhs1[-G:] = 0
        #left column
        rhs1[::G] = 0
        #right column
        rhs1[(G-1)::G] = 0

        return rhs1

    def step(self,N,ustar,vstar): #later add factors potentially
        b = self.generateb(N,ustar,vstar)
        b = np.append(b,0)
        #print(np.linalg.matrix_rank(self.A))
        #print(b)

        

        return sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(self.A_wlagrangian),b)[:-1]

N=3
k=10

p = pressureCalc(N,k)

class pressureStep():
    def __init__(self,N):
        self.N = N
        self.h = 1/N

    def step(self,N,k,ustar,vstar,phinplus1):
        G = N+2
        Gx = gradientoperator.phigradientX(G,self.h)
        Gy = gradientoperator.phigradientY(G,self.h)

        g_outputx = np.zeros((G,G+1))#even though the gradient operator only outputs the interior points it doesnt matter
        g_outputx[1:-1,1:-1] = Gx.applyandinteriorize(phinplus1).reshape((N,N+1))
        g_outputy = np.zeros((G+1,G))#because on the boundary the pressure gradient is 0 as established by the BCs in calculating phin+1
        g_outputy[1:-1,1:-1] = Gy.applyandinteriorize(phinplus1).reshape((N+1,N))

        #print(g_outputx)
        #print(g_outputy)

        #print(phinplus1.reshape((G,G)))

        #print(Gy.applyandinteriorize(phinplus1).reshape((N+1,N)))

        u_nplus1 = ustar.reshape((G,G+1)) #left most column of useless ghost points is excluded in the sum because pressure grad is not defined there
        #print(u_nplus1)
        u_nplus1 = u_nplus1 - k*g_outputx

        v_nplus1 = vstar.reshape((G+1,G)) #bottom most row of useless ghost points is excluded in the sum because pressure grad is not defined there
        #print(v_nplus1)
        v_nplus1 = v_nplus1 - k*g_outputy

        return u_nplus1.flatten(),v_nplus1.flatten()
