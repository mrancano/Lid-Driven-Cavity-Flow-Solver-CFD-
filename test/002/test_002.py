#testing unpacking and plotting utils
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src import meshgen

N = 10

utils = meshgen.meshGenClass(N)

##TEST FOR U unpack
#generate sample solution  T_sol Arranged : T_1,1 T_2,1 ... T_N_x,1 T_1,2 T_2,2 .... T_N_x,N_y
'''
u_vec = np.zeros((1,(N+1)*(N)))

i=0
for y in utils.y_vecdual:
    for x in utils.x_vec:
        u_vec[0,i] = x**2-y**2
        i+=1
U_matrix = utils.unpackUvec(u_vec)

print(U_matrix[1,0])

hdual = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(utils.x_meshU,utils.y_meshU,U_matrix,50)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''
'''
##TEST for V unpack
#generate sample solution  T_sol Arranged : T_1,1 T_2,1 ... T_N_x,1 T_1,2 T_2,2 .... T_N_x,N_y
v_vec = np.zeros((1,(N+1)*(N)))

i=0
for y in utils.y_vec:
    for x in utils.x_vecdual:
        v_vec[0,i] = x**2-y**2
        i+=1
V_matrix = utils.unpackVvec(v_vec)

print(V_matrix[1,0])

hdual = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(utils.x_meshV,utils.y_meshV,V_matrix,50)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''

##TEST for phi unpack
#generate sample solution  T_sol Arranged : T_1,1 T_2,1 ... T_N_x,1 T_1,2 T_2,2 .... T_N_x,N_y
phi_vec = np.zeros((1,(N)*(N)))

i=0
for y in utils.y_vecdual:
    for x in utils.x_vecdual:
        phi_vec[0,i] = x**2-y**2
        i+=1
phi_matrix = utils.unpackphivec(phi_vec)

print(phi_matrix[1,0])

hdual = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(utils.x_meshdual,utils.y_meshdual,phi_matrix,50)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
