import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.linalg as slg

x=np.linspace(0,2,3)
y=np.arange(2,-1,-1)
z=np.linspace(1,1000,1000)
Y,Z,X=np.meshgrid(y,z,x)

fig = plt.figure(figsize=(7,6))     # open a new figure
ax=p3.Axes3D(fig) 
surf = ax.plot_surface(X[0],Y[0],Z[0])
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
ax.set_zlabel(r'+$z\longrightarrow$')
plt.title('Plot of Z = 1 plane')

fig = plt.figure(figsize=(7,6))
ax=p3.Axes3D(fig) 
surf = ax.plot_surface(X[:,0,:],Y[:,0,:],Z[:,0,:])
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
ax.set_zlabel(r'+$z\longrightarrow$')
plt.title('Plot of Y = 2 plane')

fig = plt.figure(figsize=(7,6))
ax=p3.Axes3D(fig)  
surf = ax.plot_surface(X[:,:,0],Y[:,:,0],Z[:,:,0])
plt.xlabel(r'+ $x\longrightarrow$')
plt.ylabel(r'+ $y\longrightarrow$')
ax.set_zlabel(r'+$z\longrightarrow$')
plt.title('Plot of X = 0 plane')

plt.show()