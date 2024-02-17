import numpy as np
import scipy.linalg as sp
import os 
import sys
import matplotlib.pyplot as plt
import convectiveoperator,solutiontools,meshgen,laplacianoperator


#input N (changing to input N)
def Vstarstep(N,k,Re,u_n,v_n,u_nminus1,v_nminus1): #we have to assume that inputs are including ghost points with appropriate boundary
    G = N+2 #ghost point size
    N_v = convectiveoperator.VconvectiveOperator(G)        #conditions already applied
    L = laplacianoperator.Vlaplacian(G)
    vg = solutiontools.utilsClass(G)
    u_hatn = vg.uinterp(u_n)
    u_hatnminus1 = vg.uinterp(u_nminus1)

    rhs1 = k/2*(3*N_v.apply(u_hatn,v_n)-N_v.apply(u_hatnminus1,v_nminus1))+k/(2*Re)*L.applyVlap(v_n)
    rhs1mat = np.zeros((G+1,G))#for N = 3 the G is 5, valid domain is 4 by 3, and including ghost is 6 by 5
    rhs1mat[1:-1,1:-1]= np.reshape(rhs1,((N+1,N)))#for G 5 valid domain is 4 by 3
    

    rhs = u_n+rhs1mat.flatten()


    

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
    rhs[:,::G] = 0

    
    
    #apply right boundary on rows of matrix for ghost points knowing that v is 0 on boundary so we take v_ghost as - the point to the left
    # rows are matrix with 1 on diag and then another 1 away (which is the one to the left) size of matrix is (G+1) rows by (G+1)*G
    rightmatrix = np.eye((G+1)*G)
    np.fill_diagonal(rightmatrix[1:,:],1)
    A_pre[G-1::G] = rightmatrix[G-1::G]
    rhs[:,G-1::G] = 0

    
    

    #apply bottom boundary on rows of matrix for points ON the bottom boundary (setting 1 on main diag and 0 on rhs)
    A_pre[G:2*G] = np.eye((G+1)*G)[G:2*G]
    rhs[:,G:2*G] = 0
    
    #apply top boundary on rows of matrix for point ON the right boundary (setting 1 on main diag and 0 on rhs)
    A_pre[-2*G:-G] = np.eye((G+1)*G)[(-2*G):-G]
    rhs[:,(-2*G):-G]= 0

    

    #apply bottom boundary on rows of matrix for useless ghost points setting entire row to zero (and rhs)
    A_pre[:G] = 0
    rhs[:,:G] = 0
    #apply right boundary on rows of matrix for useless ghost points setting entire row to zero (and rhs)
    A_pre[-G:] = 0
    rhs[:,-G:] = 0

    #you can print A_pre to verify
    print(rhs)
    

    return rhs

N = 3
G = N+2
#L = laplacianoperator.Ulaplacian(N)
ug = solutiontools.utilsClass(G)

u_vec,v_vec = ug.genSol()

result = Vstarstep(N,1,1,u_vec,v_vec,np.ones(((G+1)*G,1)).T,np.zeros(((G+1)*G,1)).T)

#print(result)
#print(L.ULap)
#print(np.shape(L.ULap))
#print(np.eye(np.shape(L.ULap)[0]))
