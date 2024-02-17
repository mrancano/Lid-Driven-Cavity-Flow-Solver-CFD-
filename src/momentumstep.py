import numpy as np
import scipy as sp
import os 
import sys
import matplotlib.pyplot as plt
import convectiveoperator,solutiontools,meshgen,laplacianoperator
#STILL HAVE TO SET 1 ON DIAGONAL FOR USELESS POINTS #AND SORT OUT DIVISION BY h^2 for Laplace and by 2h for N operator
class Vstarstep():
    def __init__(self,N,k,Re):
        G = N+2
        self.h = 1/N
        L = laplacianoperator.Vlaplacian(G,self.h)
        A_pre = np.eye(len(L.VLap))-k/(2*Re)*L.VLap #without BCs but of the size including the ghost points
                                                    #we are going to have to set some rows of the matrix to calculate BCs (g can be set on rhs)
                                                    #rows of matrix correspond to rows of rhs the ones that are zero need BCs
                                                    #maybe this can be used to access those rows easier in the A matrix construction
                                                    #This means that for the Laplacian the size needs to be (N+2)i.e G so the calculations
                                                    #for the valid points work given a u_star that includes the ghost points

        #apply left boundary on rows of matrix for ghost points knowing that v is 0 on boundary so we take v_ghost as - the point to the right
        # rows are matrix with 1 on diag and then another 1 away (which is the one to the right) size of matrix is (G+1) rows by (G+1)*G
        leftmatrix = np.eye((G+1)*G)
        np.fill_diagonal(leftmatrix[:,1:],1)
        A_pre[::G] = leftmatrix[::G]

        #apply right boundary on rows of matrix for ghost points knowing that v is 0 on boundary so we take v_ghost as - the point to the left
        # rows are matrix with 1 on diag and then another 1 away (which is the one to the left) size of matrix is (G+1) rows by (G+1)*G
        rightmatrix = np.eye((G+1)*G)
        np.fill_diagonal(rightmatrix[1:,:],1)
        A_pre[G-1::G] = rightmatrix[G-1::G]

        #apply bottom boundary on rows of matrix for points ON the bottom boundary (setting 1 on main diag and 0 on rhs)
        A_pre[G:2*G] = np.eye((G+1)*G)[G:2*G]
        
        #apply top boundary on rows of matrix for point ON the right boundary (setting 1 on main diag and 0 on rhs)
        A_pre[-2*G:-G] = np.eye((G+1)*G)[(-2*G):-G]

        #apply bottom boundary on rows of matrix for useless ghost points (setting 1 on main diag and 0 on rhs)
        A_pre[:G] = np.eye((G+1)*G)[:G]
        
        #apply right boundary on rows of matrix for useless ghost points (setting 1 on main diag and 0 on rhs)
        A_pre[-G:] = np.eye((G+1)*G)[-G:]

        self.A = A_pre

        #print(np.linalg.matrix_rank(self.A))

    def generateb(self,N,k,Re,u_n,v_n,u_nminus1,v_nminus1): #we have to assume that inputs are including ghost points with appropriate boundary
        G = N+2 #ghost point size
        h = 1/N
        N_v = convectiveoperator.VconvectiveOperator(G,h)        #conditions already applied
        L = laplacianoperator.Vlaplacian(G,h)
        vg = solutiontools.utilsClass(G)
        u_hatn = vg.uinterp(u_n)
        u_hatnminus1 = vg.uinterp(u_nminus1)

        rhs1 = k/2*(3*N_v.apply(u_hatn,v_n)-N_v.apply(u_hatnminus1,v_nminus1))+k/(2*Re)*L.applyVlap(v_n)
        rhs1mat = np.zeros((G+1,G))#for N = 3 the G is 5, valid domain is 4 by 3, and including ghost is 6 by 5
        rhs1mat[1:-1,1:-1]= np.reshape(rhs1,((N+1,N)))#for G 5 valid domain is 4 by 3
        
        
        rhs = v_n+rhs1mat.flatten()
        #left boundary on rhs
        rhs[::G] = 0     
        
        #right boundary on rhs
        rhs[G-1::G] = 0

        #apply bottom boundary on rhs for points ON the bottom boundary 
        rhs[G:2*G] = 0
        
        #apply top boundary on rhs for point ON the right boundary
        rhs[(-2*G):-G]= 0
        #apply bottom boundary on rhs for useless ghost points
        rhs[:G] = 0
        #apply top boundary on rhs for useless ghost points
        rhs[-G:] = 0

        return rhs
    
    def step(self,N,k,Re,u_n,v_n,u_nminus1,v_nminus1):
        b = self.generateb(N,k,Re,u_n,v_n,u_nminus1,v_nminus1)

        

        return sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(self.A),b)
 #STILL HAVE TO SET 1 ON DIAGONAL FOR USELESS POINTS   
class Ustarstep():
    def __init__(self,N,k,Re):
        G = N+2 #ghost point size
        self.h = 1/N
        L = laplacianoperator.Ulaplacian(G,self.h)

        A_pre = np.eye(len(L.ULap))-k/(2*Re)*L.ULap #without BCs but of the size including the ghost points
                                                    #we are going to have to set some rows of the matrix to calculate BCs (g can be set on rhs)
                                                    #rows of matrix correspond to rows of rhs the ones that are zero need BCs
                                                    #maybe this can be used to access those rows easier in the A matrix construction
                                                    #This means that for the Laplacian the size needs to be (N+2)i.e G so the calculations
                                                    #for the valid points work given a u_star that includes the ghost points

        #apply bottom boundary on rows of matrix for ghost points knowing that u is 0 on boundary so we take u_ghost as - the point above
        # rows are matrix with 1 on diag and then another 1 G away (which is the one above) size of matrix is (G+1) rows by (G+1)*G
        bottomboundarymat = np.zeros(((G+1),(G+1)*G))
        # Set 1 on the main diagonal
        np.fill_diagonal(bottomboundarymat[:, :], 1)
        # Set 1 on the diagonal representing point directly above
        np.fill_diagonal(bottomboundarymat[:, (G+1):], 1)
        #apply BCs to mat and to RHS
        A_pre[:(G+1),:] = bottomboundarymat

        #apply top boundary on rows of matrix for ghost points knowing that u is 1 on boundary so we take u_ghost as 2*velocity - the point above
        # rows are matrix with 1 on diag and then another 1 G away (which is the one above) size of matrix is (G+1) rows by (G+1)*G
        topboundarymat = np.zeros(((G+1)*G,(G+1)*G))
        # Set 1 on the main diagonal
        np.fill_diagonal(topboundarymat[:, :], 1)
        # Set 1 on the diagonal representing point directly below
        np.fill_diagonal(topboundarymat[(G+1):, :], 1)
        #apply BCs to mat and to RHS
        A_pre[-(G+1):,:] = topboundarymat[-(G+1):,:]
        #apply left boundary on rows of matrix for points ON the left boundary (setting 1 on main diag and 0 on rhs)
        A_pre[(G+2)::(G+1)] = np.eye((G+1)*G)[(G+2)::(G+1)]
        #apply right boundary on rows of matrix for point ON the right boundary (setting 1 on main diag and 0 on rhs)
        A_pre[(2*G)::(G+1)] = np.eye((G+1)*G)[(2*G)::(G+1)]
        #apply left boundary on rows of matrix for useless ghost points setting entire row to zero (and rhs)
        A_pre[::(G+1)] = np.eye((G+1)*G)[::(G+1)]
        #apply right boundary on rows of matrix for useless ghost points setting entire row to zero (and rhs)
        A_pre[(G)::(G+1)] = np.eye((G+1)*G)[G::(G+1)]

        self.A = A_pre

        

    def generateb(self,N,k,Re,u_n,v_n,u_nminus1,v_nminus1): #we have to assume that inputs are including ghost points with appropriate boundary
        G = N+2 #ghost point size
        h = 1/N
        N_u = convectiveoperator.UconvectiveOperator(G,h)        #conditions already applied
        L = laplacianoperator.Ulaplacian(G,h)
        ug = solutiontools.utilsClass(G)
        v_hatn = ug.vinterp(v_n)
        v_hatnminus1 = ug.vinterp(v_nminus1)

        rhs1 = k/2*(3*N_u.apply(u_n,v_hatn)-N_u.apply(u_nminus1,v_hatnminus1))+k/(2*Re)*L.applyUlap(u_n)
        rhs1mat = np.zeros((G,G+1))#for N = 3 the G is 5, valid domain is 3 by 4, and including ghost is 5 by 6
        rhs1mat[1:-1,1:-1]= np.reshape(rhs1,((N,N+1)))#for G 5 valid domain is 3 by 4
        

        rhs = (u_n+rhs1mat.flatten()).T


        #ORDER OF BCs matter if you want to make any of the numbers non-zero as there are intersection points

        

        #top ghost boundary (here is velocity impact u_ghost = 2-u_above)
        rhs[-(G+1):] = 2


        #left boundary (useless)
        rhs[::(G+1)] = 0     
        
        #right boundary (useless)
        rhs[G::(G+1)] = 0

        #left boundary (on boundary)
        rhs[1::(G+1)] = 0     
        
        #right boundary (on boundary)
        rhs[(G-1)::(G+1)] = 0

        #bottom ghost boundary
        rhs[:(G+1)] = 0
        
        return rhs
    
    def step(self,N,k,Re,u_n,v_n,u_nminus1,v_nminus1):
        b = self.generateb(N,k,Re,u_n,v_n,u_nminus1,v_nminus1)

        #print(b)
        #print(len(b))
        #print(np.linalg.matrix_rank(self.A))
        #print(np.shape(self.A))

        return sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(self.A),b.flatten())
    

N = 3
G = N+2
k = 1
Re = 1

x = Vstarstep(N,k,Re)

ug = solutiontools.utilsClass(G)

u_vec,v_vec = ug.genSol()

#print(x.generateb(N,k,Re,u_vec,v_vec,np.ones(((G+1)*G,1)).T,np.zeros(((G+1)*G,1)).T))
#print(x.A)
