import numpy as np
import scipy.linalg as sp
import os 
import sys
import matplotlib.pyplot as plt
import convectiveoperator,solutiontools,meshgen,laplacianoperator


#input N+2 (changing to input N)
def Ustarstep(N,k,Re,u_n,v_n,u_nminus1,v_nminus1): #we have to assume that inputs are including ghost points with appropriate boundary
    G = N+2 #ghost point size
    N_u = convectiveoperator.UconvectiveOperator(G)        #conditions already applied
    L = laplacianoperator.Ulaplacian(G)
    ug = solutiontools.utilsClass(G)
    v_hatn = ug.vinterp(v_n)
    v_hatnminus1 = ug.vinterp(v_nminus1)

    rhs1 = k/2*(3*N_u.apply(u_n,v_hatn)-N_u.apply(u_nminus1,v_hatnminus1))+k/(2*Re)*L.applyUlap(u_n)
    rhs1mat = np.zeros((G,G+1))#for N = 3 the G is 5, valid domain is 3 by 4, and including ghost is 5 by 6
    rhs1mat[1:-1,1:-1]= np.reshape(rhs1,((N,N+1)))#for G 5 valid domain is 3 by 4
    

    rhs = u_n+rhs1mat.flatten()


    

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
    rhs[:,:(G+1)] = 0 #not including pressure gradients for now

    #apply top boundary on rows of matrix for ghost points knowing that u is 1 on boundary so we take u_ghost as 2*v - the point above
    # rows are matrix with 1 on diag and then another 1 G away (which is the one above) size of matrix is (G+1) rows by (G+1)*G
    topboundarymat = np.zeros(((G+1)*G,(G+1)*G))
    # Set 1 on the main diagonal
    np.fill_diagonal(topboundarymat[:, :], 1)
    # Set 1 on the diagonal representing point directly below
    np.fill_diagonal(topboundarymat[(G+1):, :], 1)
    #apply BCs to mat and to RHS
    A_pre[-(G+1):,:] = topboundarymat[-(G+1):,:]
    rhs[:,-(G+1):] = 2 #consider general velocity term do later (setting pressure gradients to zero for now)


    #apply left boundary on rows of matrix for points ON the left boundary (setting 1 on main diag and 0 on rhs)
    A_pre[(G+2)::(G+1)] = np.eye((G+1)*G)[(G+2)::(G+1)]
    rhs[:,(G+2)::(G+1)] = 0
    
    #apply right boundary on rows of matrix for point ON the right boundary (setting 1 on main diag and 0 on rhs)
    A_pre[(2*G)::(G+1)] = np.eye((G+1)*G)[(2*G)::(G+1)]
    rhs[:,(2*G)::(G+1)] = 0

    #apply left boundary on rows of matrix for useless ghost points setting entire row to zero (and rhs)
    A_pre[::(G+1)] = 0
    rhs[:,::(G+1)] = 0
    #apply right boundary on rows of matrix for useless ghost points setting entire row to zero (and rhs)
    A_pre[(G)::(G+1)] = 0
    rhs[:,(G)::(G+1)] = 0

    #you can print A_pre to verify
    
    print(A_pre)

    return rhs

N = 3
G = N+2
#L = laplacianoperator.Ulaplacian(N)
ug = solutiontools.utilsClass(G)

u_vec,v_vec = ug.genSol()

result = Ustarstep(N,1,1,u_vec,v_vec,np.ones(((G+1)*G,1)).T,np.zeros(((G+1)*G,1)).T)

#print(result)
#print(L.ULap)
#print(np.shape(L.ULap))
#print(np.eye(np.shape(L.ULap)[0]))


    