import numpy as np
import matplotlib.pyplot as plt
import meshgen

def plotx(N,u_n,v_n,phi,Re,t,f,CFL):
    G = N+2
    m = meshgen.meshGenClass(N)
    f = round(f,3)

    X = m.x_meshdual
    Y = m.y_meshdual

    XU = m.x_meshU
    YU = m.y_meshU

    pressure = phi.reshape((G,G))[1:-1,1:-1]
    x_vel = u_n.reshape((G,G+1))[1:-1,1:-1]

    plt.contourf(XU, YU, x_vel, cmap='viridis', levels=20)
    #plt.colorbar(label='Pressure')
    plt.colorbar(label='x_vel')

    # Set axis labels
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Set title
    title_string = f"N = {N} | Re = {Re} | t = {t} | f = {f}|CFL={CFL}"
    plt.title(title_string)

    # Show the plot
    plt.show()

def ploty(N,u_n,v_n,phi,Re,t,f,CFL):
    G = N+2
    m = meshgen.meshGenClass(N)
    f = round(f,3)
    X = m.x_meshdual
    Y = m.y_meshdual

    XV = m.x_meshV
    YV = m.y_meshV

    #pressure = phi.reshape((G,G))[1:-1,1:-1]
    y_vel = v_n.reshape((G+1,G))[1:-1,1:-1]

    plt.contourf(XV, YV, y_vel, cmap='viridis', levels=20)
    #plt.colorbar(label='Pressure')
    plt.colorbar(label='y_vel')

    # Set axis labels
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Set title
    title_string = f"N = {N} | Re = {Re} | t = {t} | f = {f}|CFL={CFL}"
    plt.title(title_string)

    # Show the plot
    plt.show()

def plotp(N,u_n,v_n,phi,Re,t,f,CFL):
    G = N+2
    m = meshgen.meshGenClass(N)
    f = round(f,3)
    X = m.x_meshdual
    Y = m.y_meshdual


    pressure = phi.reshape((G,G))[1:-1,1:-1]
    #y_vel = v_n.reshape((G,G+1))[1:-1,1:-1]

    plt.contourf(X, Y, pressure, cmap='viridis', levels=20)
    #plt.colorbar(label='Pressure')
    plt.colorbar(label='Pressure (phi)')

    # Set axis labels
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Set title
    title_string = f"N = {N} | Re = {Re} | t = {t}|f = {f}|CFL={CFL}"
    plt.title(title_string)

    # Show the plot
    plt.show()
