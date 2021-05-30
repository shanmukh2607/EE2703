"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Final Examination 
    Solution
    Finding the Magnetic Field for a Loop Antenna

Usage : $python3 <filename.py>

Pseudo Code:

IMPORT required modules
SET x,y,z arrays
COMPUTE X,Y,Z meshgrids
SET Radius and Number of sections in loop
DETERMINE angular positions of section on Loop
COMPUTE position vectors of section on Loop
DEFINE current elements on loop for a section
CALL current elements for all sections
COMPUTE tangential vectors for each section on loop
DEFINE CALC function:
    INPUT section index -l
    RETURN  A due to l-th section
INITIALISE A
FOR all sections:
    INCREMENT A by CALC(l-th section)
COMPUTE B
PLOT B vs z
COMPUTE a least squares fit for B vs z
PLOT Least squares fit
"""

############################ CODE BEGINS HERE ############################################

"""
First we import the required modules
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slg

"""
Breaking the Volume into 3 x 3 x 1000 grid
We choose the space:
x belongs to [-1,1], y belongs to [-1,1] and z belongs to [1,1000]
Each point separated by a 1cm apart (1 unit) in grid
"""
x=np.linspace(0,2,3)
y=np.arange(2,-1,-1)
z=np.linspace(1,1000,1000)

Y,Z,X=np.meshgrid(y,z,x)  
# Meshgrid function to create X,Y,Z grids in the mentioned space

"""
Loop Segments, Position vectors and current elements
"""
rad = 10                   # Radius
N = 100                    # No. of sections
midpt_angles = np.linspace(0,2*np.pi,N+1)[:-1] #Angular positions of midpoints of each segment
coses = np.cos(midpt_angles)
sines = np.sin(midpt_angles)
x_c = rad*coses # Coordinates of ...
y_c = rad*sines # ... Midpoints

"""
pts: Position vector for all points
pts[l] : Position vector for l-th segment
"""
pts = np.vstack((x_c,y_c)).T   # pts is a 100x2 shaped postion vector

#Plotting midpoint postions
fig,ax = plt.subplots(figsize=(6,6))
plt.xlabel(r'+ $x$ (in cm)$\longrightarrow$')
plt.ylabel(r'+ $y$ (in cm)$\longrightarrow$')
plt.plot(pts[:,0],pts[:,1],'ro',label = 'Midpoint of Element')
plt.legend()
plt.title('Midpoint of Elements')
plt.grid()

def currentelement_spatial(phi,dl):
    """
    Returns current element vector for given angular position phi of segment
    4pi/mu_0 = 1e7
    """
    return dl*np.array([-1e7*np.cos(phi)*np.sin(phi),1e7*np.cos(phi)**2]).T

dl = 2*np.pi*rad/N                       # Magnitude of dl = r*d_theta
Idl = currentelement_spatial(midpt_angles,dl)   # Function call
fig,ax = plt.subplots(figsize=(6,6))
ax.quiver(pts[:,0],pts[:,1],Idl[:,0],Idl[:,1],color='b')
plt.xlabel(r'+ $x$ (in cm)$\longrightarrow$')
plt.ylabel(r'+ $y$ (in cm)$\longrightarrow$')
plt.title('Current Elements in a Loop Antenna')
plt.grid()


dl_vec = dl*np.vstack((-sines,coses)).T

def calc(l):
    """
    Input : l lies between 0 to N-1
    Return :
    This function returns the vector A field for all points in grid
    due to the l-th segment in the loop.
    """
    coords = pts[l]  # Midpoint of l-th segment
    xl = coords[0]   # x coordinate
    yl = coords[1]   # y coordinate
    x_diff = X - xl  
    y_diff = Y - yl
    distance = np.sqrt(x_diff**2 + y_diff**2 + Z**2)  # L2 distance
    distance = distance.reshape((1000,3,3,1))  
    # Reshaping the array to get the advantage of broadcasting
    return coses[l]*np.exp(-1j*0.1*distance)*dl_vec[l]/distance

"""
Finding net A vector due to all segments in the loop
"""
A = 0   #Initialising variable
for l in range(N):
    A += calc(l)   #Increment
    
"""
Find magnetic field along +z direction using A vector
"""
B = (A[:,1,2,1] - A[:,1,0,1] + A[:,2,1,0] - A[:,0,1,0])/2

# Log-log plot of magnitude of B vs z
fig,ax = plt.subplots(figsize=(7,6))
plt.loglog(z,np.abs(B),'bo-',markersize=3)
plt.xlabel(r'+ $z$ (in cm)$\longrightarrow$')
plt.ylabel(r'+ $B_z$ (in Tesla) $\longrightarrow$')
plt.title('Log-Log plot of Magnetic Field along +z direction')
plt.grid()

"""
Using Least squares method to fit the data to
B = c*z**b
logB = logc + b*z
logc and b are the parameters to be estimated using Least Squares method
"""
C = np.log(np.abs(B))
A = np.c_[np.ones(1000),np.log(z)]
params = slg.lstsq(A,C)[0]   # Lstsq fit parameters

c = np.exp(params[0])
b = params[1]
print('The following parameters are estimated by Least Squares Method')
print('b is ',b)
print('c is ',c)

# Plotting the least squares fit
B_est = c*z**b

fig,ax = plt.subplots(figsize=(7,6))
plt.loglog(z,np.abs(B),'bo-',label='Original',markersize=3)
plt.loglog(z,B_est,'ro-',label='Least Squares Fit',markersize=2)
plt.xlabel(r'+ $z$ (in cm)$\longrightarrow$')
plt.ylabel(r'+ $B_z$ (in Tesla)$\longrightarrow$')
plt.title('Log-Log plot of Least Squares Fit vs Original Field')
plt.legend()
plt.grid()

plt.show()
############################ CODE ENDS HERE ###################################