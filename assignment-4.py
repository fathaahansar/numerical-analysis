# Solution of 2D Poisson's equation with FVM
# M.F.H. Ansarul Huq (c) 2021

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

Lx = 10 # length in x direction
Ly = 5 # length in y direction
Nx = 200 # number of intervals in x-direction
Ny = 100 # number of intervals in y-direction
dx = Lx/Nx # grid step in x-direction
dy = Ly/Ny # grid step in y-direction

# Defining the given source function f(x,y)
def sourceF(x,y):
    F = 0; alpha = 40
    for i in range(1,10):
        for j in range(1,5):
            F = F + np.exp(-alpha * pow((x-i),2) - alpha * pow((y-j),2))
    return F

# Defining a coefficient function k(x,y) = 1
def coeffK1(x,y):
    K = 1.0
    return K

# Defining the given coefficient function k(x,y)
def coeffK(x,y):
    K = 1 + 0.1 * (x + y + x*y)
    return K

# Defining a function to create an lexicographic array of f values for the given domain and step size
def createF(funcName):
    Fvec = []
    for j in range(0,Ny+1):
        for i in range(0,Nx+1):
            Fvec += [funcName(i*dx,j*dy)]
    return Fvec

# Defining a function to create an lexicographic array of inner f values for the given domain and step size
def createF_inner(funcName):
    Fvec = []
    for j in range(1,Ny):
        for i in range(1,Nx):
            Fvec += [funcName(i*dx,j*dy)]
    return Fvec

# Function to create a system matrix for solving the linear algebraic FVM equation
def create2DLFVM(coeffFun):

    D1 = []  # Mid diagonal
    D2 = []  # Off diagonal 1
    D3 = []  # Off diagonal 2

    for j in range(1, Ny):
        for i in range(1, Nx):
            D1 += [coeffFun(dx * (i - 0.5), dy * j) / dx ** 2 + coeffFun(dx * i, dy * (j - 0.5)) / dy ** 2 + coeffFun(
                dx * (i + 0.5), dy * j) / dx ** 2 + coeffFun(dx * i, dy * (j + 0.5)) / dy ** 2]
            if np.size(D2) != ((Nx - 1) * (Ny - 1) - 1):
                if i == Nx - 1:
                    D2 += [0]
                else:
                    D2 += [-coeffFun(dx * (i + 0.5), dy * j) / dx ** 2]

            if j != Ny - 1:
                D3 += [-coeffFun(dx * i, dy * (j + 0.5)) / dy ** 2]

    A = sp.diags([D1, D2, D2, D3, D3], [0, -1, 1, -(Nx-1), Nx-1], format='csc')

    return A

# Using the function defined to generate lexicographic 1-D array of f and k
fvec = createF(sourceF)
kvec = createF(coeffK)

# Reshape to a 2-D matrix
fmat = np.reshape(fvec,(Ny+1,Nx+1))
kmat = np.reshape(kvec,(Ny+1,Nx+1))

# Visualizing the source function f
plt.clf()
plt.figure(1)
plt.title("Source Function f")
plt.imshow(fmat,origin='lower',extent=(0,Lx,0,Ly))
plt.colorbar(orientation='horizontal')
plt.savefig("f_plot_"+str(Nx)+"x"+str(Ny)+".png")

# Visualizing the coefficient function k
plt.clf()
plt.figure(2)
plt.title("Coefficient Function k")
plt.imshow(kmat,origin='lower',extent=(0,Lx,0,Ly))
plt.colorbar(orientation='horizontal')
plt.savefig("k_plot_"+str(Nx)+"x"+str(Ny)+".png")

# Visualize the system matrix for k = 1
#print(create2DLFVM(coeffK1).toarray())

A = create2DLFVM(coeffK)
f_inner = createF_inner(sourceF)

# Visualize the system matrix for k(x,y)
#print(A.toarray())

# Solving the linear algebra problem
u = la.spsolve(A, f_inner)

# Visualize the lexicographic solution array for u(x,y)
print(u)

ufill = np.reshape(u,(Ny-1,Nx-1))

# Visualize the solution matrix u(x,y)
plt.clf()
plt.figure(3)
plt.title("Solution u")
plt.imshow(ufill, origin='lower', extent=(0, Lx, 0, Ly))  # use the ufill array here
plt.colorbar(orientation='horizontal')
plt.savefig("u_plot_"+str(Nx)+"x"+str(Ny)+".png")
