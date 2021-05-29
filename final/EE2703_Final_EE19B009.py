"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Final Examination 
    Solution
    Finding the Magnetic Field for a Loop Antenna

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
x=np.linspace(-1,1,3)
y=np.arange(1,-2,-1)
z=np.linspace(1,1000,1000)

Y,Z,X=np.meshgrid(y,z,x)  
# Meshgrid function to create X,Y,Z grids in the mentioned space

"""
Loop Segments, Position vectors and current elements
"""
rad = 10                   # Radius
N = 100                    # No. of sections
midpt_angles = np.linspace(0,2*np.pi,N+1)[:-1]  #Angular positions of midpoints of each segment
x_c = rad*np.cos(midpt_angles) # Coordinates of ...
y_c = rad*np.sin(midpt_angles) # ... Midpoints

"""
pts: Position vector for all points
pts[l] : Position vector for l-th segment
"""
pts = np.vstack((x_c,y_c)).T   # pts is a 100x2 shaped postion vector

#Plotting midpoint postions
fig,ax = plt.subplots(figsize=(6,6))
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
plt.plot(pts[:,0],pts[:,1],'ro',label = 'Midpoint of Element')
plt.legend()
plt.title('Midpoint of Elements')
plt.grid()

def current_spatial(x,y):
    """
    Returns current element vector for given index(l) of segment
    4pi/mu_0 = 1e7
    """
    return np.array([-1e7*x*y/(x**2 + y**2),1e7*x**2/(x**2 + y**2)])

I = current_spatial(pts[:,0],pts[:,1])   # Function call
dl = 2*np.pi*rad/N                       # Magnitude of dl = r*d_theta
fig,ax = plt.subplots(figsize=(6,6))
ax.quiver(pts[:,0],pts[:,1],I[0]*dl,I[1]*dl,color='b')
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
plt.title('Current Elements in a Loop Antenna')
plt.grid()


dl_vec = dl*np.vstack((-y_c/np.sqrt(x_c**2 + y_c**2),x_c/np.sqrt(x_c**2 + y_c**2))).T

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
    return xl*np.exp(-1j*0.1*distance)*dl_vec[l]/distance/np.sqrt(xl**2 + yl**2)

"""
Finding net A vector due to all segments in the loop
"""
A = 0   #Initialising variable
for l in range(N):
    A += calc(l)   #Increment
    
"""
Find magnetic field along +z direction using A vector
"""
B = (A[:,1,2,1] - A[:,1,0,1] + A[:,2,1,0] - A[:,0,1,0])/4

# Log-log plot of magnitude of B vs z
fig,ax = plt.subplots(figsize=(7,6))
plt.loglog(z,np.abs(B))
plt.xlabel(r'+ $z\longrightarrow$')
plt.ylabel(r'+ $B_z\longrightarrow$')
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
print('b is ',b)
print('c is ',c)

# Plotting the least squares fit
B_est = c*z**b

fig,ax = plt.subplots(figsize=(7,6))
plt.loglog(z,np.abs(B),'bo',label='Original',markersize=3)
plt.loglog(z,B_est,'ro',label='Lstsq Fit',markersize=2)
plt.xlabel(r'+ $z\longrightarrow$')
plt.ylabel(r'+ $B_z\longrightarrow$')
plt.title('Log log plot of Least squares fit vs Original Field')
plt.legend()
plt.grid()

plt.show()