# Solution of 2D Poisson's equation with FDM
# Your M.F.H. Ansarul Huq (c) 2021

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

# Defining all the functions that are used
def FDLaplacian2D(dx,dy,Nx,Ny):
    Dx = (1 / dx) * np.delete(sp.diags([[1 for i in range(Nx)], [-1 for i in range(Nx)]], [0, -1]).toarray(), Nx - 1, 1)
    Dy = (1 / dy) * np.delete(sp.diags([[1 for i in range(Ny)], [-1 for i in range(Ny)]], [0, -1]).toarray(), Ny - 1, 1)

    DxT = Dx.transpose()
    DyT = Dy.transpose()

    Lxx = DxT.dot(Dx)
    Lyy = DyT.dot(Dy)

    Ix = sp.eye(Nx - 1).toarray()
    Iy = sp.eye(Ny - 1).toarray()

    return(sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix))

def sourcefunc(x,y):
    return (1 + x + y - x*y)

def domainfunc(x,y):
    return (pow(pow(x,2)+pow(y,2)-1, 3) - pow(x,2)*pow(y,3))

# Starting number of sub-intervals
LeftX = -1.5
RightX = 1.5
LeftY = -1.5
RightY = 1.5

Nx = 100 # number of intervals in x-direction
Ny = 100 # number of intervals in y-direction
dx = (RightX-LeftX)/Nx  # grid step in x-direction
dy = (RightY-LeftY)/Ny  # grid step in y-direct

x,y = np.mgrid[LeftX+dx:RightX:dx, LeftY+dy:RightY:dy]

x = np.transpose(x); y = np.transpose(y)

f = sourcefunc(x,y)

domain = domainfunc(x,y)

rows,cols,vals = sp.find(domain<0)

minc = np.min(f) # background color
ffill = minc*np.ones([Ny-1,Nx-1]) # empty rectangular image
ffill[rows,cols] = f[rows,cols] # filling part of the image ffill with the values of f over deformed domain

plt.ion()
plt.figure(1)
plt.clf()
plt.title("Source Function f in the Rectangular and Given Domain")

plt.subplot(1,2,1)
plt.imshow(f, origin='lower', extent=(LeftX, RightX, LeftY, RightY))  # use the f array here
plt.colorbar(orientation='horizontal')

plt.subplot(1,2,2)
plt.imshow(ffill, origin='lower', extent=(LeftX, RightX, LeftY, RightY))  # use ffill array here
plt.colorbar(orientation='horizontal')

plt.savefig("f_domain_plot_100x100.png")

# lexicographic domain function
domainLX = np.reshape(domain, domain.size)  # reshape 2D domain array into 1D domainLX array

# find lexicographic indices of inner points
rowsLX,colsLX,valsLX = sp.find(domainLX<0)

# lexicographic source vector on rectangular domain
fLX = np.reshape(f, f.size, order='F')  # reshape 2D f array into 1D fLX array

# lexicographic source vector on deformed domain
fLXd = fLX[colsLX]

# 2D FD Laplacian on rectangular domain
A = FDLaplacian2D(dx,dy,Nx,Ny)

# 2D FD Laplacian on deformed domain
Ad = A.tocsr()[colsLX,:].tocsc()[:,colsLX]

u = la.spsolve(Ad, fLXd)

# preparing to display the solution
minc = np.min(u) # background color
ufill = minc*np.ones([Ny-1,Nx-1]) # empty rectangular image
ufill[rows,cols] = u # filling part of the image with the solution

plt.ion()
plt.figure(2)
plt.clf()

plt.imshow(ufill, origin='lower', extent=(LeftX, RightX, LeftY, RightY))  # use the ufill array here
plt.colorbar(orientation='horizontal')
plt.savefig("u_plot_100x100.png")

# Calculating the Eigenvalues and Eigenvectors of Matrix A
eigenvalues_A, eigenvectors_A = la.eigsh(A, k=20, which='SM')
eigenvalues_Ad, eigenvectors_Ad = la.eigsh(Ad, k=20, which='SM')

# solving for u using eigenvectors and values: u = V.Inv(Lambda).Transpose(V).V_10
u_v10 = np.dot(np.dot(eigenvectors_Ad * sp.diags(1/eigenvalues_Ad), eigenvectors_Ad.transpose()), eigenvectors_Ad[:, 9])

minc = np.min(u_v10) # background color
ufill = minc*np.ones([Ny-1,Nx-1]) # empty rectangular image
ufill[rows,cols] = u_v10 # filling part of the image with the solution

plt.ion()
plt.figure(3)
plt.clf()

plt.imshow(ufill, origin='lower', extent=(LeftX, RightX, LeftY, RightY))
plt.colorbar(orientation='horizontal')
plt.savefig("u_f=v10_100x100.png")

print(sp.diags(eigenvalues_Ad))
print(eigenvectors_A)