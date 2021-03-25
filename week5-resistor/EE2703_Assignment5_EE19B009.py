"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Assignment 5 Solution
    The Resistor Problem
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.linalg as slg

"""
This block of code make to correct input from the command line and raises a customized message in case of error
In case no input parameters are passed, the code takes default parameters as shown below
"""

if len(sys.argv)==5:
    try:
        Nx = int(sys.argv[1])
        Ny = int(sys.argv[2])
        radius = float(sys.argv[3])
        Niter = int(sys.argv[4])
        
    except ValueError:
        print("Input arguments must be integers. Radius can be float too")
        exit()

elif len(sys.argv)==1:
    Nx = 25
    Ny = 25
    radius = 8
    Niter = 1500
    print("No input parameters entered. Default parameters are chosen")

else:
    print('\nExpected Usage: $python3 %s <Nx> <Ny> <Radius> <No. of iterations>' % sys.argv[0])
    exit()

"""
Print out the values entered by the user/ default values
"""
print('Nx is',Nx)
print('Ny is',Ny)
print('radius is',radius)
print('No of iterations is',Niter)

"""
Initialization
Creating a 2D array for potential and a meshgrid for visualization purposes
"""
phi = np.zeros((Ny,Nx)) 
#No of columns is Nx and No of rows is Ny
x = np.arange(Nx) - (Nx-1)/2  
# x contains the discrete positions along X direction (right)
y = np.flip(np.arange(Ny) - (Ny-1)/2) 
# y contains the discrete positions along Y direction (top)
X,Y = np.meshgrid(x,y)
ii = np.where(X*X + Y*Y <= radius**2)  
# indices /postions of points which lie within the radius
phi[ii] =1 # These set of points always remain at potential = 1

"""
Plot for Initial Potential
"""
fig,ax = plt.subplots(figsize=(6,6),num =0)
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
ax.contour(X,Y,phi)
ax.scatter(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,marker = 'o', color ='r')
plt.title(r'Contour plot of Initial Potential')

"""
Finding phi in an iterative method using the Laplace Equation in difference method
Also recording absolute error in each iteration
"""
errors = np.zeros(Niter)
for k in range(Niter):
    oldphi = phi.copy() # saving a copy 
    phi[1:-1,1:-1] = 0.25*(phi[1:-1,0:-2] + phi[1:-1,2:] + phi[0:-2,1:-1] + phi[2:,1:-1]) 
    # estimating phi from laplace equation
    phi[ii] =1  # restoring the condition of Potential =1 inside circle
    phi[1:-1,0] = phi[1:-1,1]  # Boundary conditions
    phi[1:-1,-1] = phi[1:-1,-2] # On hanging
    phi[0,1:-1] = phi[1,1:-1] #Sides
    errors[k] = np.max(np.abs(phi-oldphi)) # error

"""
Using Least Square Method (LSTSQ) to get parameters for best fit curve for
y = A*exp(Bx) or
logy = logA + B*x

M = [1 x]
v = [p1 p2]T
c = [logy]

Here our parameters p1 and p2 are logA and B respectively 
""" 
i = np.arange(1,Niter+1) 
one_array = np.ones(Niter) # array with all 1s
M = np.c_[one_array,i]   # for all Niter
M_500 = np.c_[one_array[500:],i[500:]] # from 500th iter
v = slg.lstsq(M,np.log(errors))[0]    #Least Squares
v_500 = slg.lstsq(M_500,np.log(errors[500:]))[0]

"""
Semilog Plot of Error
"""
fig,ax = plt.subplots(num =1)
plt.semilogy(i,errors)
plt.xlabel(r'No. of iterations') # Labels
plt.ylabel(r'Error')
plt.title(r'Error in semilog plot') #Title
plt.grid() # Grid

"""
Log Log plot of Error along with plot of every 50th iter
"""
fig,ax = plt.subplots(num =2)
ax.loglog(i,errors, label='Error')
ax.loglog(i[::50],errors[::50], 'ro',label ='Every 50th iter')
plt.xlabel(r'No. of iterations') # Labels
plt.ylabel(r'Error')
plt.title(r'Error in log-log plot') #Title
plt.legend() #Legend
plt.grid() #Grid

"""
Error using best fit parameters for all iterations
and Error using best fit parameter from 500th iter
"""
error_lstsq = np.exp(np.dot(M,v))
error500_lstsq = np.exp(np.dot(M_500,v_500))

"""
Comparing the actual error with lstsq predicted error
"""
fig,ax = plt.subplots(num =3)
ax.loglog(errors, label='Iterative Error (True Error)')
ax.loglog(i[::50],error_lstsq[::50],'ro', label = 'Least Sqaures Fit')
ax.loglog(i[500::50],error500_lstsq[::50],'go', label = 'Least Sqaures Fit from 500th iteration')
plt.xlabel(r'No. of iterations') #Labels
plt.ylabel(r'Error')
plt.title(r'Best fit for Error in log-log plot') #Title
plt.legend() #Legend
plt.grid() #Grid

"""
v is the solution of lstsq and the parameters are logA and B
"""
A = np.exp(v[0])
B = v[1]

def cum_error(x):
    """
    Input argument x: No. of iterations
    Output is cumulative max possible error
    """
    return -A/B*np.exp(B*(x+0.5))

"""
Log log plot for cumulative error
"""
fig, ax = plt.subplots(num =4)
ax.loglog(i[100::100],cum_error(i[100::100]),'ro')
plt.xlabel(r'No. of iterations')
plt.ylabel(r'Cumulative Error')
plt.title(r'Cumulative Error in log-log plot (Every 100th iter)')
plt.grid()

"""
Contour plot of potential
"""
fig,ax = plt.subplots(figsize=(6,6),num =5)
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
plt.title(r'Contour Plot of Potential $\phi$')
cs = ax.contour(X,Y,phi)
ax.scatter(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,marker = 'o', color ='r')
ax.clabel(cs, inline =1, fontsize = 9) 

"""
Surface plot of potential
"""
fig1=plt.figure(6)     # open a new figure
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
plt.title('The 3-D surface plot of the potential')
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
surf = ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=plt.cm.jet)


"""
Computing Current density J from the equations
"""
Jx = 0.5*(phi[1:-1,:-2]-phi[1:-1,2:]) # 0.5*(phi(x-1,y) - phi(x+1,y)) in cartesian
Jy = 0.5*(phi[2:,1:-1]-phi[:-2,1:-1]) # 0.5*(phi(x, y-1) - phi(x,y+1)) in cartesian

"""
Plot for Current flow
"""
fig,ax = plt.subplots(figsize=(6,6),num =7)
ax.quiver(X[1:-1,1:-1],Y[1:-1,1:-1],Jx,Jy)
ax.scatter(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,marker = 'o', color ='r')
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
plt.title('Vector plot of current flow')
plt.show()

