"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Assignment 6 Solution
    The Tubelight Problem
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import *


def probability_check(x):
    """
    To check for allowed values of probability
    """
    try:
        p = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid input: %r is not a floating point value"%x)
    if p<0.0 or p>1.0 :
        raise argparse.ArgumentTypeError("%r is not a valid probability value. Must lie between [0,1]"%x)
    return p

def natural_check(x):
    """
    To check if input values are natural numbers
    """
    try:
        n = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid input:%r is not an integer."%x)
    if n<=0:
        raise argparse.ArgumentTypeError("%r is not a positive integer. Please enter a positive integer"%n)
    return n

def positive_check(x):
    """
    To check if input values are positive numbers
    """
    try:
        y = float(x)
    except ValueError:
        raise argparse.ArgumentErrorType("Invalid input:%r is not a floating point value"%x)
    if y<=0:
        raise argparse.ArgumentErrorType("%r is not a positive value. Please enter a positive value"%y)
    return y

#Collecting input by parsing command line arg
parser = argparse.ArgumentParser()
parser.add_argument("--n", type = natural_check, default = 100, help = 'Spatial Grid size')
parser.add_argument("--M", type = natural_check, default = 5, help = 'No. of electrons injected per turn')
parser.add_argument("--nk", type = natural_check, default = 500, help = 'No. of turns to simulate')
parser.add_argument("--u0", type = positive_check, default = 5, help = 'Threshold velocity')
parser.add_argument("--p", type = probability_check, default = 0.25, help = 'Probability that ionization will occur')
parser.add_argument("--Msig", type= positive_check, default = 2, help = 'Standard Deviation of Electron distribution')
args = vars(parser.parse_args())
n,M,nk,u0,p,Msig=(args['n'],args['M'],args['nk'],args['u0'], args['p'], args['Msig'])

# Tubelight Simulation Code
xx = np.zeros(n*M) # Initialisation of position array
u = np.zeros(n*M)  # velocity array
dx = np.zeros(n*M) # displacement array
I = []
X = []             # Empty lists for collecting Intensity, Position and Velocity data for all runs
V = []

for k in range(nk):     # Loop through each turn of simulation
    N = int(Msig*np.random.randn() + M)  # No. of electrons in the grid
    # Injection of e- at cathode
    start = np.where(xx==0)
    xx[start[0][:N]] = 1
    # Propagation : Update the prev position and velocity and compute
    ii = np.where(xx>0)
    X.extend(xx[ii].tolist())
    V.extend(u[ii].tolist())
    dx[ii] = u[ii] + 0.5
    u[ii] +=1
    xx[ii] += dx[ii]
    # Reset the status of e- which reached end of grid
    end = np.where(xx>=n)
    xx[end] = 0
    u[end] = 0
    dx[end] = 0
    # Threshold 
    kk = np.where(u>= u0)
    ll = np.where(np.random.rand(len(kk[0]))<=p)
    kl = kk[0][ll]
    # position update of ionized e-
    # dt is uniformly sampled between [0,1] and 
    # xx is recorrected for ionized e- by using dt
    dt = np.random.rand()
    xx[kl] = (xx[kl] - dx[kl]) + (u[kl]-1)*dt + 0.5*dt**2
    # Velocity update of ionized e-
    u[kl] = 0
    I.extend(xx[kl].tolist())

#Plots
fig, ax = plt.subplots(num=0,figsize=(7, 5))
plt.hist(X,bins=n,edgecolor='black')
plt.xlabel('Position')
plt.ylabel('No. of electrons')
plt.title(r'Electron Density')

fig, ax = plt.subplots(num=1,figsize=(7, 5))
count, bins, rect =plt.hist(I,bins=n,range=[0,n],edgecolor='black')
plt.xlabel('Position')
plt.ylabel('No. of ionized electrons')
plt.title(r'Intensity of Emitted Light')

fig, ax = plt.subplots(num=2,figsize=(7,7))
plt.plot(X,V,'rx',markersize =5)
plt.xlabel(r' $Position\longrightarrow$')
plt.ylabel(r' $Velocity\longrightarrow$')
plt.title('Electron Phase Space')

#intensity data
xpos=0.5*(bins[0:-1]+bins[1:])
with open('data.txt','w+') as f:
    print("Intensity Data",file=f)
    print(tabulate(np.stack((xpos,count)).T,["xpos","count"]),file=f)
print('Intensity data printed in data.txt file')
plt.show()