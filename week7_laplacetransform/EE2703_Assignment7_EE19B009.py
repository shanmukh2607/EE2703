"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Assignment 7 Solution
    The Laplace Transform
"""
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

#Problem 1 and 2
def spring1(decay):
    """
    A function that inputs decay and returns the impulse response of the system
    """
    n1 = np.poly1d([1,decay])
    d1 = np.poly1d([1,decay])**2 + 2.25    
    d2 = np.poly1d([1,0,2.25])
    H1 = sp.lti(n1,d1*d2)     # Laplace Transform of output signal
    t,h= sp.impulse(H1,None,np.linspace(0,50,1001))
    return t,h
t1, h1 = spring1(0.5)
t2, h2 = spring1(0.05)


fig,ax = plt.subplots(num=1,figsize=(7,5))
ax.plot(t1,h1,'r')
plt.xlabel(r' $t\longrightarrow$')
plt.ylabel(r' $x\longrightarrow$')
plt.grid()
plt.title('Time response of a Spring for decay=0.5',size=15)


fig,ax = plt.subplots(num=2,figsize=(7,5))
ax.plot(t2,h2,'b')
plt.xlabel(r' $t\longrightarrow$')
plt.ylabel(r' $x\longrightarrow$')
plt.grid()
plt.title('Time response of a Spring for decay=0.05',size=15)

#Problem 3
fig,ax = plt.subplots(num=3,figsize=(9,7))
H = sp.lti(1,[1,0,2.25])   #Defining transfer function of system
t = np.linspace(0,50,1001)
w = np.arange(1.4,1.6,0.05)
for w0 in w:
    f = np.cos(w0*t)*np.exp(-0.05*t)
    t,y,temp =sp.lsim(H,f,t)    #Finding the response for input signals with various frequencies
    ax.plot(t,y,label='Freq. = %1.2f'%w0)
plt.legend()
plt.xlabel(r' $t\longrightarrow$')
plt.ylabel(r' $x\longrightarrow$')
plt.grid()
plt.title('System response for various frequencies')

fig,ax=plt.subplots(num=4,figsize=(7,5))
w,s,phi = H.bode()
plt.semilogx(w,s)
plt.xlabel(r' $\omega\longrightarrow$')
plt.ylabel(r' Magnitude in dB$\longrightarrow$')
plt.grid()
plt.title('Magnitude Response of Spring system')

fig,ax=plt.subplots(num=5,figsize=(7,5))
plt.semilogx(w,phi)
plt.xlabel(r' $\omega\longrightarrow$')
plt.ylabel(r' $\phi\longrightarrow$')
plt.grid()
plt.title('Phase Response of Spring system')

#Question 4 Coupled spring
X = sp.lti([1,0,2],[1,0,3,0]) # Laplace transform of x(t)
Y = sp.lti([2],[1,0,3,0])     #Laplace tranform of y(t)
t,x=sp.impulse(X,None,np.linspace(0,20,201))  # x(t)
t,y=sp.impulse(Y,None,np.linspace(0,20,201))  # y(t)

fig,ax = plt.subplots(num=6,figsize=(9,7))
plt.plot(t,x,'r')
plt.plot(t,y,'b')
plt.legend(["$x(t)$", "$y(t)$"])
plt.xlabel(r' $t\longrightarrow$')
plt.grid()
plt.title(r'$x(t)$ and $y(t)$ in Coupled spring system')

#Problem 5 RLC ciruit
R = 100 
L = 1e-6
C = 1e-6
Hrlc = sp.lti(1,[L*C,R*C,1])  # Transfer function of system
w,s,phi = Hrlc.bode()

fig,ax=plt.subplots(num=7,figsize=(7,5))
plt.semilogx(w,s)
plt.xlabel(r' $\omega\longrightarrow$')
plt.ylabel(r' Magnitude in dB$\longrightarrow$')
plt.grid()
plt.title('Magnitude Response')

fig,ax=plt.subplots(num=8,figsize=(7,5))
plt.semilogx(w,phi)
plt.xlabel(r' $\omega\longrightarrow$')
plt.ylabel(r' $\phi\longrightarrow$')
plt.grid()
plt.title('Phase Response')

#Problem 6
def v_in(t):
    """
    Returns a signal for a given time interval
    """
    return np.cos(1e3*t) - np.cos(1e6*t)
t_small = np.linspace(0,3e-5,61) #Upto 30 us
t_big = np.linspace(0,1e-2,1001) #Upto 10ms
t_small,y_small,temp =sp.lsim(Hrlc,v_in(t_small),t_small)
t_big,y_big,temp =sp.lsim(Hrlc,v_in(t_big),t_big)

fig,ax=plt.subplots(num=9,figsize=(7,5))
ax.plot(t_small,y_small,'red')
plt.xlabel(r' $t\longrightarrow$')
plt.ylabel(r' $x\longrightarrow$')
plt.grid()
plt.title(r"Response of Two port RLC Network for 30 $\mu s$")

fig,ax=plt.subplots(num=10,figsize=(7,5))
ax.plot(t_big,y_big,'red')
plt.xlabel(r' $t\longrightarrow$')
plt.ylabel(r' $x\longrightarrow$')
plt.grid()
plt.title(r"Response of Two port RLC Network for 10 $ms$")
plt.show()