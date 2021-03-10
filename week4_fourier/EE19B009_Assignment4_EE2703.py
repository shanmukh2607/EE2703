"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Assignment 4 Solution
    Fourier Approximations
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.linalg as linalg

"""
All required functions are defined as written below
"""

def exp(x):
    """
    Exponential function defined as exp(x) where x is a vector (or scalar)
    """
    return np.exp(x)

def coscos(x):
    """
    cos(cos(x)) takes input argument as x which is a vector (or scalar)
    """
    return np.cos(np.cos(x))

def periodic_maker(fun):
    """
    periodic_maker is a function that takes input argument as function and outputs a function which is 2pi periodic
    i.e function value between [0,2pi) is repeated on the entire real line
    """
    def newfunc(x):
        return fun(np.remainder(x,2*np.pi))
    return newfunc

def u(x,k,f):
    """
    u is useful for finding Fourier coefficients by integration method
    """
    return f(x)*np.cos(k*x)

def v(x,k,f):
    """
    v is useful for finding Fourier coefficients by integration method
    """
    return f(x)*np.sin(k*x)

def fourier(func, n):
    """
    fourier function computes 'n' fourier coefficients, n necessarily being odd for the function 'func'
    by intergration method
    It returns a one dimensional vector of fourier coefficients
    """
    f_vec = np.zeros(n) #initialization
    
    f_vec[0] = integrate.quad(func, 0, 2*np.pi)[0]/(2*np.pi) #integrating to find a0
    
    for i in range(1,(n+1)//2):
        f_vec[2*i-1] = integrate.quad(u, 0, 2*np.pi, args=(i,func))[0]/np.pi #to find ak
        f_vec[2*i] = integrate.quad(v, 0, 2*np.pi, args=(i,func))[0]/np.pi # to find bk
    
    return f_vec 

def fourier_lstsq(func,x):
    """
    fourier_lstsq computes 51 fourier coefficients by the method of Least Squares.
    it takes function 'func' and array of x values,'x' as inputs
    It returns a one dimensional vector of fourier coefficients and the fourier approximation values of f(x) for given x
    """
    b = func(x) #function values assigned to vector b
    A = np.zeros((x.size,51)) #initaializing A matrix of shape (Number of x points)x51
    A[:,0] =1 # Initializing first column with 1s
    for k in range(1,26):
        A[:,2*k-1] = np.cos(k*x) #odd indexed columns with cos(kx)
        A[:,2*k] = np.sin(k*x) #even indexed columns with sin(kx)
    c = linalg.lstsq(A,b)[0] #computing the best fit 'c' vector
    f_est = np.dot(A,c) #fourier approximation values of f(x)
    return c, f_est

"""
End of defining all functions required
Main code implementation starts from here
"""

exp_periodic = periodic_maker(exp)  # exp_periodic is the expected fourier series function for exp(x)
coscos_periodic = periodic_maker(coscos) #coscos_periodic is the expected fourier series function for cos(cos(x))
x = np.linspace(-2*np.pi, 4*np.pi, 10000) # x is numpy array with 10k points linearly spaced between -2pi and 4pi

"""
Semilogy Plot of exp(x) and it's expected Fourier series function
"""

fig, ax = plt.subplots(num=1)
ax.semilogy(x,exp(x),label ='Actual Function')
ax.semilogy(x,exp_periodic(x), 'r--', label='Periodic Extension')
plt.xlabel(r'$x \longrightarrow$')
plt.ylabel(r'$e^x \longrightarrow$')
plt.title(r'Semilogy plot of $e^x$ vs $x$ in $[-2\pi,4\pi)$')
plt.legend()
plt.grid()

"""
Plot of cos(cos(x)) and it's expected Fourier series function
"""

fig, ax = plt.subplots(figsize=(7,5),num=2)
ax.plot(x, coscos(x), label ='Actual Function')
ax.plot(x, coscos_periodic(x), 'r--',label='Periodic Extension')
plt.xlabel(r'$x \longrightarrow$')
plt.ylabel(r'$cos(cos(x)) \longrightarrow$')
plt.title(r'Plot of $cos(cos(x))$ vs $x$ in $[-2\pi,4\pi)$')
plt.legend()
plt.grid()

"""
Finding fourier coefficients by integration and least squares method
"""
# 0 to 2pi 
x = np.linspace(0,2*np.pi,401)
x = x[:-1]

#Calling integration method
exp_fvec = fourier(exp, 51)
coscos_fvec = fourier(coscos, 51)

# Calling least squares method
(exp_lstsq, exp_est) = fourier_lstsq(exp,x)
(coscos_lstsq, coscos_est) = fourier_lstsq(coscos,x)

#Question 6: max deviation of fourier coefficients
print('Largest deviation in Coefficients of')
print('exp(x) is',np.amax(np.abs(exp_fvec - exp_lstsq)))
print('cos(cos(x)) is',np.amax(np.abs(coscos_fvec - coscos_lstsq)))

"""
Semilog Plot of fourier coefficients of exp(x) by method of integration and least squares
Coefficients by integration method plotted in RED
Coefficients by least squares method plotted in GREEN
"""

fig, ax = plt.subplots(num=3)
ax.semilogy(np.abs(exp_fvec),'ro', label = 'Coefficients by Integration')
ax.semilogy(np.abs(exp_lstsq),'go', label = 'Coefficients by Least Squares', markersize= 4.5)
plt.xlabel(r'$n \longrightarrow$')
plt.ylabel(r'$Magnitude \longrightarrow$')
plt.title(r'Semilogy plot of Fourier coefficients of $e^x$ ')
plt.legend()
plt.grid()

"""
Log Log Plot of fourier coefficients of exp(x) by method of integration and least squares
Coefficients by integration method plotted in RED
Coefficients by least squares method plotted in GREEN
"""

fig, ax = plt.subplots(num=4)
ax.loglog(np.abs(exp_fvec),'ro',label = 'Coefficients by Integration')
ax.loglog(np.abs(exp_lstsq),'go', label = 'Coefficients by Least Squares', markersize =4.5)
plt.xlabel(r'$n \longrightarrow$')
plt.ylabel(r'$Magnitude \longrightarrow$')
plt.title(r'Log-Log plot of Fourier coefficients of $e^x$ ')
plt.legend()
plt.grid()

"""
Semilog Plot of fourier coefficients of cos(cos(x)) by method of integration and least squares
Coefficients by integration method plotted in RED
Coefficients by least squares method plotted in GREEN
"""

fig, ax = plt.subplots(num=5)
ax.semilogy(np.abs(coscos_fvec),'ro', label = 'Coefficients by Integration')
ax.semilogy(np.abs(coscos_lstsq), 'go', label = 'Coefficients by Least Squares', markersize =4.5)
plt.xlabel(r'$n \longrightarrow$')
plt.ylabel(r'$Magnitude \longrightarrow$')
plt.title(r'Semilogy plot of Fourier coefficients of $cos(cos(x))$')
plt.legend()
plt.grid()

"""
Log Log Plot of fourier coefficients of cos(cos(x)) by method of integration and least squares
Coefficients by integration method plotted in RED
Coefficients by least squares method plotted in GREEN
"""

fig, ax = plt.subplots(num=6)
ax.loglog(np.abs(coscos_fvec),'ro', label ='Coefficients by Integration')
ax.loglog(np.abs(coscos_lstsq), 'go',  label = 'Coefficients by Least Squares', markersize =4.5)
plt.xlabel(r'$n \longrightarrow$')
plt.ylabel(r'$Magnitude \longrightarrow$')
plt.title(r'Log-Log plot of Fourier coefficients of $cos(cos(x))$ ')
plt.legend()
plt.grid()

"""
Plot of fourier approximation (by least squares method) of exp(x)
along with actual function value
"""

fig, ax = plt.subplots(num=7)
ax.semilogy(x,exp_est, 'go', label='Using Least squares', markersize =3)
ax.semilogy(x,exp(x),'r',label ='Actual Function')
plt.xlabel(r'$x \longrightarrow$')
plt.ylabel(r'$e^x \longrightarrow$')
plt.title(r'Semilogy plot of $e^x$ vs $x$ in $[0,2\pi)$')
plt.legend()
plt.grid()

"""
Plot of fourier approximation (by least squares method) of cos(cos(x)) 
along with actual function value
"""

fig, ax = plt.subplots(num=8)
ax.plot(x,coscos_est, 'go', label='Using Least Squares',markersize = 3)
ax.plot(x,coscos(x),'r',label ='Actual Function')
plt.xlabel(r'$x \longrightarrow$')
plt.ylabel(r'$cos(cos(x)) \longrightarrow$')
plt.title(r'Semilogy plot of $cos(cos(x))$ vs $x$ in $[0,2\pi)$')
plt.legend()
plt.grid()

plt.show()
# END of CODE