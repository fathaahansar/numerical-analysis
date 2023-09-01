# Solution of 1D Poisson's equation with FDM
# M.F.H. Ansarul Huq (c) 2021

import numpy as np
import matplotlib.pyplot as plt

# Starting number of sub-intervals
n = 5
n_limit = 200

# For studying the rate of convergence of FDM, defining the x-axis coordinates
n_set = np.linspace(5, n_limit, n_limit-4)

# Boundary values for both u1 and u2
u12_0 = -6
u12_n = -2

# Initiating error data set for plotting the convergence
e1_array = np.zeros(n_limit-4); e2_array = np.zeros(n_limit-4); e1_array_normal = np.zeros(n_limit-4)

# Defining the 2 source functions to compute for any x
def func1(x):
    return (-6 * x + 2)

def func2(x):
    return (-12 * x * x - 6 * x + 2)

# Defining function to calculate exact solution of u1 and u2 at any x
def func_u1(x):
    return (pow(x,3) - pow(x,2) + x - 3)

def func_u2(x):
    return (pow(x,4) + pow(x,3) - pow(x,2) + x - 4)

# Iterating through n for Problem 7, where we have to understand convergence of FDM
while n <= n_limit:

    # Vector x with values
    xgrid = np.linspace(-1.0,1.0, n+1)

    # Uncomment for Intermediate Value Extraction
    #print("xgrid")
    #print(xgrid)

    # Calculate step size
    h = (xgrid[n]-xgrid[0])/n

    # Uncomment for Intermediate Value Extraction
    #print("h")
    #print(h)

    # Compute the value of functions f1 and f2 at the grid points
    f1 = func1(xgrid)
    f2 = func2(xgrid)

    # Uncomment for Intermediate Value Extraction
    #/print("f1")
    #print(f1)
    #print("f2")
    #print(f2)

    # Plotting the functions f1 and f2 for n=5 only
    if n == 5:
        plt.clf()
        plt.figure()
        plt.plot(xgrid, f1, label='f_1(x)', color='blue')
        plt.plot(xgrid, f2, label='f_2(x)', color='red')
        plt.legend()
        plt.title("f_i(x) vs x")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.savefig("./Final_Plots/f1_f2_vs_x_n="+str(n)+".png")

    # Compute the exact solutions of u1 and u2 at the grid points
    u1ex = func_u1(xgrid)
    u2ex = func_u2(xgrid)

    # Uncomment for Intermediate Value Extraction
    #print("u1ex")
    #print(u1ex)

    # Uncomment for Intermediate Value Extraction
    #print("u2ex")
    #print(u2ex)

    # Constructing diagonal of Matrix A
    d1 = [2 for i in range(n-1)]; d2 = [-1 for i in range(n-2)]

    diag_1 = np.diag(d1)
    diag_2 = np.diag(d2, k=1)
    diag_3 = np.diag(d2, k=-1)

    A = (diag_1 + diag_2 + diag_3)/np.float_power(h,[2])

    # Uncomment for Intermediate Value Extraction
    #print("A")
    #print(A)

    # Plotting the matrix A for n=5 only
    if n == 5:
        plt.clf()
        plt.spy(A, marker='*', color='b')
        plt.savefig("matrix_a_n="+str(n)+".png")

    # Compute the eigenvalues and eigenvectors of Matrix A
    e_values, e_vectors = np.linalg.eig(A)

    # Output eigenvalues
    #print("Eigenvalues of Matrix A with n="+str(n)+":")
    #print(e_values)

    # Constructing the f1rhs and f2rhs
    f1rhs = [f1[i+1] for i in range(n-1)]
    f2rhs = [f2[i+1] for i in range(n-1)]

    f1rhs[0] += u12_0/pow(h,2); f1rhs[n-2] += u12_n/pow(h,2)
    f2rhs[0] += u12_0/pow(h,2); f2rhs[n-2] += u12_n/pow(h,2)

    # Uncomment for Intermediate Value Extraction
    #print("f1rhs")
    #print(f1rhs)

    # Solving the Linear Algebra problem
    u_1 = np.linalg.solve(A, f1rhs)
    u_2 = np.linalg.solve(A, f2rhs)

    # Uncomment for Intermediate Value Extraction
    #print("u_1")
    #print(u_1)

    # Uncomment for Intermediate Value Extraction
    #print("u_2")
    #print(u_2)

    # Type casting for appending boundary values
    u_1_inner = list (u_1)
    u_2_inner = list (u_2)

    # Uncomment for Intermediate Value Extraction
    #print("u_1_inner")
    #print(u_1_inner)

    # Uncomment for Intermediate Value Extraction
    #print("u_2_inner")
    #print(u_2_inner)

    u_1_full = []; u_2_full = []
    i = 0

    # Constructing the full u_i array
    while i <= n:
        if i == 0:
            u_1_full.append(u12_0)
            u_2_full.append(u12_0)
        elif  i == n:
            u_1_full.append(u12_n)
            u_2_full.append(u12_n)
        else:
            u_1_full.append(u_1_inner[i-1])
            u_2_full.append(u_2_inner[i-1])
        i += 1

    # Uncomment for Intermediate Value Extraction
    #print("u_1_full")
    #print(u_1_full)
    #print("u_2_full")
    #print(u_2_full)

    # Plotting u1 and u2 along with their exact solution curves for n=5 only
    if n == 5:
        plt.clf()
        plt.plot(xgrid, u1ex, label='u_1ex(x)', color='blue')
        plt.plot(xgrid, u2ex, label='u_2ex(x)', color='red')
        plt.plot(xgrid, u_1_full, label='u_1(x)', color='blue', linestyle='dashed')
        plt.plot(xgrid, u_2_full, label='u_2(x)', color='red', linestyle='dashed')
        plt.legend()
        plt.title("Comparison between the Exact and FDM solutions")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.savefig("u(x)_vs_x_n="+str(n)+".png")

    # Error Calculation for the functions u1 and u2
    e1 = u1ex - u_1_full
    e2 = u2ex - u_2_full

    # Calculating the RMSE (Global Error)
    power_array = [2 for i in range(n+1)]
    e1_array[n-5] = np.log(pow(np.sum(np.float_power(e1,power_array))/(n-1),0.5))
    e2_array[n-5] = np.log(pow(np.sum(np.float_power(e2,power_array))/(n-1),0.5))

    e1_array_normal[n-5] = pow(np.sum(np.float_power(e1,power_array))/(n-1),0.5)

    # Output the global error values
    #print("Global Error for u1 for n="+str(n))
    #print(pow(np.sum(np.float_power(e1,power_array))/(n-1),0.5))

    #print("Global Error for u2 for n="+str(n))
    #print(pow(np.sum(np.float_power(e2,power_array))/(n-1),0.5))

    n += 1

# Plotting convergence for u1
plt.clf()
plt.plot(n_set, e1_array, label='log(Global Error) u1', color='blue')
plt.legend()
plt.title("Global Error Convergence of u1")
plt.xlabel("n")
plt.ylabel("log(Global Error)")
plt.savefig("u1globalerror_vs_n_nmax="+str(n_limit)+".png")

# Plotting convergence for u2
plt.clf()
plt.plot(n_set, e2_array, label='log(Global Error) u2', color='red')
plt.legend()
plt.title("Global Error Convergence of u2")
plt.xlabel("n")
plt.ylabel("log(Global Error)")
plt.savefig("u2globalerror_vs_n_nmax="+str(n_limit)+".png")





