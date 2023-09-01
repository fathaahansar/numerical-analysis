# Modeling atmospheric pollution with the Finite-Element Method
# M.F.H. Ansarul Huq (c) 2021

from __future__ import print_function
from dolfin import *

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

# Time step definition
T = 10
num_steps = 20
dt = T/num_steps

# Rectangle domain with mesh
Lx = Ly = 10
nx = ny = 100

mesh = RectangleMesh(Point(0,0),Point(Lx,Ly),nx,ny)

# Finite element with Lagrange basis function
P1 = FiniteElement('Lagrange',triangle,1)

# Function space V for scalar functions
V = FunctionSpace(mesh, P1)

# Function space W for vector functions
W = FunctionSpace(mesh, MixedElement([P1,P1]))

# Velocity vector field
B = 0.5; omega = (2*np.pi)/(4*T)
velocity = Expression(('B*cos(omega*t)','B*sin(omega*t)'),degree=2,B=B,omega=omega,t=0.0)
w = Function(W)
w.interpolate(velocity)

# Define Boundary Condition (Given u=0 at all times at the boundary)
u_D = Constant(0.0)

# Dirichlet (left) boundary
def boundary(x,on_boundary):
    return on_boundary

# The essential boundary condition
bc = DirichletBC(V, u_D, boundary)

# Initial value of u_k
u_k = interpolate(u_D, V)

# Source function
A = 5
alpha = 50
x_1,x_2,x_3 = [1,1.5,4] # Given in Table 1
y_1,y_2,y_3 = [2,2.5,3] # Given in Table 1

# Variational problem
u = TrialFunction(V)
v = TestFunction(V) # Automatically assumes zero Dirichlet BC at V
f = Expression('A * exp(-alpha*(x[0]-x_1)*(x[0]-x_1) - alpha*(x[1]-y_1)*(x[1]-y_1)) + A * exp(-alpha*(x[0]-x_2)*(x[0]-x_2) - alpha*(x[1]-y_2)*(x[1]-y_2)) + A * exp(-alpha*(x[0]-x_3)*(x[0]-x_3) - alpha*(x[1]-y_3)*(x[1]-y_3))',degree=2,alpha=alpha,A=A,x_1=x_1,y_1=y_1,x_2=x_2,y_2=y_2,x_3=x_3,y_3=y_3)

D = 0.005
F = u*v*dx + D*dt*dot(grad(u),grad(v))*dx + dt*dot(w,grad(u))*v*dx - (u_k + dt*f)*v*dx

a,L = lhs(F),rhs(F)

# Time stepping loop
u = Function(V)
t = 0

# Setting up for plotting the u values
plt.ion()
plt.figure(1)
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1, num=1)
points = mesh.coordinates()

unow = u_k.compute_vertex_values(mesh)
im = ax.tripcolor(points[:,0],points[:,1],unow)
ax.set_aspect('equal', adjustable='box')
cbar = fig.colorbar(im,ax=ax, orientation='horizontal')
ax.set_title(r'$u(t)$, t = '+str(np.round(t,2)))
plt.savefig('u_plot_t='+str(t)+'.png')

unow_5 = 0; unow_10 = 0

for n in range(num_steps):

    # Updating time
    t += dt
    u_D.t = t
    velocity.t = t  # updating time inside velocity
    w.interpolate(velocity)  # updating w

    # Solve variational problem
    solve(a == L,u,bc)

    # Update previous solution
    u_k.assign(u)

    if t == 5:
        unow_5 = u_k.compute_vertex_values(mesh)
    elif t == 10:
        unow_10 = u_k.compute_vertex_values(mesh)

im.remove()
im = ax.tripcolor(points[:,0],points[:,1],unow_5)
cbar.update_normal(im)
ax.set_title(r'$u(t)$, t = 5')
plt.savefig('u_plot_t=5.png')

im.remove()
im = ax.tripcolor(points[:,0],points[:,1],unow_10)
cbar.update_normal(im)
ax.set_title(r'$u(t)$, t = 10')
plt.savefig('u_plot_t=10.png')








