"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
	EE2703 Assignment 3 Solution
    Fitting Data to a Model
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.linalg as slg
"""
The filename (or path) is given as a command line argument which can be stored into a list sys.argv. Incase more than one filename or no filename is given, it gives an expected usage message"
"""

if len(sys.argv)!=2 :
    print('\nExpected Usage: %s <inputfile>' % sys.argv[0])
    exit()
filename = sys.argv[1]


"""
The user might input a wrong file and the program gives a customized error message
"""	

try:
    time_data = np.loadtxt(filename) #time_data contains the all data 
except IOError:
    print('Invalid File')
    exit()
"""
First, let's extract all the data and plot it along with the truth value curve 
"""
t = time_data[:,0] # first col is time 
data = time_data[:,1:] # f(t) values for various t and std dev of noise
sigma = np.logspace(-1,-3,9) #std dev values for all 9 sets
sigma_str= ['%s%d = %f'%('\u03C3',i+1, sigma[i]) for i in range(len(sigma))] #preparing a string list for legend
# Plotting the data
fig, ax = plt.subplots(figsize =(10,8), num=0) #num = 0 labels it as figure 0
ax.plot(t,data)
# defining the function g
def g(t,A,B):
    return A*sp.jn(2,t) + B*t  # sp.jn gives bessel function 

y = g(t,1.05,-0.105) # function call
# Plotting True Value
ax.plot(t,y,label='True Value')
sigma_str.append('True Value')
plt.legend(sigma_str)   # Legend
plt.xlabel(r'$t \longrightarrow$',size = 15)  # Labels
plt.ylabel(r'$f(t) \longrightarrow$',size = 15)
plt.title(r'Q4 Data to be fitted to Theoertical f(t) for A = 1.05 and B = -0.105 ') # Title
plt.grid(True) # Grid 

"""
Plotting errorbars for the data in 1st column along with true values.
To avoid clustering of bars let's plot in a step of 5 points
"""

plt.figure(num=1)
figure1 = plt.gcf()
figure1.set_size_inches(10,8)
plt.errorbar(t[::5],data[::5,0],sigma[0],fmt='ro',label = 'Errorbar') #errorbars 
plt.plot(t,y,label='True Value') #true values
plt.xlabel(r'$t \longrightarrow$',size = 15)  #Labels
plt.ylabel(r'$f(t) \longrightarrow$',size = 15)
plt.legend()  # Legend
plt.title(r'Q5 Data points for $\sigma$ = 0.10 along with exact function') # Title
plt.grid(True)

"""
Understanding the function g(t;A,B) in terms of matrices
"""
A0 = 1.05
B0 = -0.105

x = sp.jn(2,t) # Bessel function
M = np.c_[x,t] 
v = np.array([
    [A0],
    [B0]  
]) # v is a column vector with parameters A0 and B0
res = np.dot(M,v).reshape(-1) 
# Matrix multiplied and Converted to a row vector
np.allclose(y, res) 
"""
Now this matrix M when matrix multiplied to a col vector [A0 B0] gives g(t,A0,B0)
"""
k =0  # column of data set as a variable for convenience
a = np.linspace(0,2,21) # selecting 21 linearly spaced A values between 0 and 2
b = np.linspace(-0.2,0,21) # similarly for B between -0.2 and 0
mse = np.zeros((21,21)) # Initialising an array
for i in range(len(a)):
    for j in range(len(b)):
        v = np.array([
            [a[i]],
            [b[j]]
        ]) # col vector [A_i B_i]
        g_vec = np.transpose(np.dot(M,v)) 
        mse[i,j]= np.sum((data[:,k]-g_vec)**2)/101 # Mean squared error for Ai and Bj values

ind = np.where(mse == np.amin(mse)) # Index at which MSE is minimum
"""
Plotting contour plot for MSE with A as X axis and B as Y axis
"""
fig ,ax = plt.subplots(figsize = (7,7),num=2)
cs = ax.contour(a,b,mse,levels = 16)  # Contour plot
ax.clabel(cs,cs.levels[:5], inline =1, fontsize = 9)  #Labeling MSE for first 5 contours
ax.plot(a[ind[0][0]],b[ind[1][0]], marker = 'o', color ='r') # Plotting the point at which min MSE occurs
ax.annotate('Location of min', xy=(a[ind[0][0]],b[ind[1][0]])) # Annotation
plt.xlabel(r'$A \longrightarrow$') #Labels
plt.ylabel(r'$B \longrightarrow$')
plt.title(r'Q8: Contour Plot of $\epsilon_{ij}$ for Column-%d data'%(k+1)) #title

"""
Plotting error in estimation of A and B using SciPy LinAlg's lstsq function w.r.t to std dev of noise in data
"""
#initialization
A_pred = np.zeros(9)
B_pred = np.zeros(9)
for k in range(0,9):
    soln =slg.lstsq(M,data[:,k])  # calling least square's function of scipy
    A_pred[k] = soln[0][0] #estimated A value for kth element in the array of std dev 
    B_pred[k] = soln[0][1] #estimated B value
A_err = np.abs(1.05 - A_pred) #Absolute value of error
B_err = np.abs(0.105 + B_pred)
fig, ax = plt.subplots()
ax.plot(sigma,A_err, 'o--',label ='$Aerr$') #Plotting error vs std dev
ax.plot(sigma,B_err, 'o--',label ='$Berr$')
plt.xlabel(r'Noise standard deviation') # Labels
plt.ylabel(r'Error in estimation')
plt.legend() # Legend
plt.title(r'Q10 Variation of Error with noise') # Title
plt.grid() # Grid

"""
Plotting error vs std dev in logarithmic scale
"""

fig, ax = plt.subplots()
ax.stem(sigma, A_err, use_line_collection = True) # stem plot is plots a vertical stem for discrete points
ax.stem(sigma, B_err, use_line_collection = True)
ax.loglog(sigma,A_err, 'ro',label ='$Aerr$') # Marked discrete points in log scale
ax.loglog(sigma,B_err, 'bo',label ='$Berr$')
plt.xlabel(r'Noise standard deviation')
plt.ylabel(r'Error in estimation') # Labels
plt.legend() # Legend
plt.title(r'Q11 Variation of Error with $\sigma$ in log scale') # Title
plt.grid() #Grid
plt.show() # Shows all the plots after running the code