"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Assignment 10 Solution
    DFT for Non-periodic signals
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as slg
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm

# Signal and Helper Functions are defined first

def sinroot2(t):
    return np.sin(np.sqrt(2)*t)

def sina(t):
    return np.sin(1.25*t)

def cos3(t):
    return np.cos(0.86*t)**3

def chirp(t):
    return np.cos(16*(1.5 + t/(2*np.pi))*t)

def dft(tstart,tend,N,f,xlim,title):
    """
    DFT spectrum without any windowing
    tstart : Start Time
    tend : End time
    N : No. of samples
    f : Signal function
    xlim : Plotting limits of x axis
    title : Title of plot
    """
    t = np.linspace(tstart,tend,N+1)
    t = t[:-1]
    dt = t[1] - t[0]
    fmax = 1/dt
    w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1)
    w = w[:-1]
    y = f(t)
    y[0] = 0
    y = np.fft.fftshift(y)
    Y = np.fft.fftshift(np.fft.fft(y))/N
    fig ,ax = plt.subplots(figsize=(7,7))
    plt.subplot(2,1,1)
    plt.plot(w,np.abs(Y),'b')
    plt.xlim([-xlim,xlim])
    plt.ylabel(r"Magnitude$\longrightarrow$")
    plt.title(r"Spectrum of " + title)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(w,np.angle(Y),'ro')
    plt.ylabel(r"$\phi$ $\longrightarrow$")
    plt.xlim([-xlim,xlim])
    plt.xlabel(r"$\omega \longrightarrow$")
    plt.grid()

def window_dft(tstart,tend,N,f,xlim,title,antisymmetry = True):
    """
    DFT spectrum with WINDOWING
    tstart : Start Time
    tend : End time
    N : No. of samples
    f : Signal function
    xlim : Plotting limits of x axis
    title : Title of plot
    antisymmetry : if true then y[0] is initialized to zero before fftshift
    """
    t = np.linspace(tstart,tend,N+1)
    t = t[:-1]
    dt = t[1] - t[0]
    fmax = 1/dt
    n=np.arange(N)
    wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
    w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1)
    w = w[:-1]
    y = f(t)*wnd
    if antisymmetry:
        y[0] = 0
    y = np.fft.fftshift(y)
    Y = np.fft.fftshift(np.fft.fft(y))/N
    fig ,ax = plt.subplots(figsize=(7,7))
    plt.subplot(2,1,1)
    plt.plot(w,np.abs(Y),'b',lw=2)
    plt.xlim([-xlim,xlim])
    plt.ylabel(r"Magnitude$\longrightarrow$")
    plt.title(r"Spectrum of "+ title +r"$\times w(t)$")
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(w,np.angle(Y),'ro')
    plt.ylabel(r"$\phi$ $\longrightarrow$")
    plt.xlim([-xlim,xlim])
    plt.xlabel(r"$\omega \longrightarrow$")
    plt.grid()
    
def estimator(y, hamming, N=128,power = 2,plotspectrum=False):
    """
    y : input vector
    hamming : If true then hamming window is applied to signal
    N : 128 samples default value
    power : default value set to 2
    plotspectrum : If false then spectrum is not plotted
    """
    t = np.linspace(-np.pi,np.pi,N+1)
    t = t[:-1]
    fmax = 1.0/(t[1]-t[0])
    n = np.arange(128)
    wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(N-1)))
    w = np.linspace(-np.pi*fmax,np.pi*fmax,N+1)
    w = w[:-1]
    if hamming:
        y = y*wnd
    y[0] = 0
    y = np.fft.fftshift(y)
    Y = np.fft.fftshift(np.fft.fft(y))/N 
    # Prediction of w
    w_pred = np.sum(np.abs(w)*np.abs(Y)**power)/np.sum(np.abs(Y)**power)
    # Prediciton of delta by least squares method
    c1 = np.cos(w_pred*t)
    c2 = np.sin(w_pred*t)
    A = np.c_[c1,c2]
    params = slg.lstsq(A, y)[0]
    cosd = params[0]
    sind = params[1]
    d_pred = np.arctan(-sind/cosd)
    #print(w_pred,d_pred)
    if plotspectrum:
        fig ,ax = plt.subplots(figsize=(7,7))
        plt.subplot(2,1,1)
        plt.plot(w,np.abs(Y),'b')
        plt.xlim([-4,4])
        plt.ylabel(r"Magnitude$\longrightarrow$")
        plt.title(r"Spectrum of y")
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(w,np.angle(Y),'ro')
        plt.ylabel(r"$\phi$ $\longrightarrow$")
        plt.xlim([-4,4])
        plt.xlabel(r"$\omega \longrightarrow$")
        plt.grid()
    return w_pred, d_pred

def y_generator(w0,d0,N=128):
    """
    This function generates a vector of signal data
    w0 : frequency of cosine signal
    d0 : phase of cosine signal
    """
    t = np.linspace(-np.pi,np.pi,N+1)
    t = t[:-1]
    return np.cos(w0*t + d0)

def y_whitenoise(w0,d0,N=128):
    """
    This function generates a vector of signal data with additional noise
    """
    t = np.linspace(-np.pi,np.pi,N+1)
    t = t[:-1]
    return np.cos(w0*t + d0) + 0.1*np.random.randn(N)    

def estimation_test(gen,w0,d0,l):
    """
    This function does the estimation of parameters for various l and gives the best figures.
    w0,d0 : true Values
    l : No of indices to be checked for estimation
    """
    index = np.linspace(1.5,3,l)
    w_error = np.zeros(l)
    d_error = np.zeros(l)
    for i in range(len(index)):
        w_pred, d_pred = estimator(gen(w0,d0),False,power = index[i])
        w_error[i] = np.abs(w0 - w_pred)
        d_error[i] = np.abs(d0 - d_pred)
    i_best = np.argmin(w_error + d_error)  
    # We achieve the best index when both errors are minimum
    w_best , d_best = estimator(gen(w0,d0),False,power = index[i_best],plotspectrum=True)
    print('Predicted w and delta of the signal are ',w_best,' and ',d_best)
    print('We achieve the best estimation at index: ',index[i_best])
    print('Error in w prediction is ',w_error[i_best])
    print('Error in delta prediction is ',d_error[i_best])
    
# Question 1 : Repeat the exercises of assignment
## 1A : Spectrum of sin(root2*t)
dft(-np.pi,np.pi,64,sinroot2,8,r'$\sin\left(\sqrt{2} t\right)$')

## 1B : Plotting the actual signal
t1=np.linspace(-np.pi,np.pi,65);
t1=t1[:-1]
t2=np.linspace(-3*np.pi,-np.pi,65);
t2=t2[:-1]
t3=np.linspace(np.pi,3*np.pi,65);
t3=t3[:-1]
fig,ax = plt.subplots(figsize=(7,6))
plt.plot(t1,np.sin(np.sqrt(2)*t1),'b',lw=2)
plt.plot(t2,np.sin(np.sqrt(2)*t2),'r',lw=2)
plt.plot(t3,np.sin(np.sqrt(2)*t3),'r',lw=2)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$")
plt.grid()

## 1C : Plotting the signal for which we computed dft
t1=np.linspace(-np.pi,np.pi,65);
t1=t1[:-1]
t2=np.linspace(-3*np.pi,-np.pi,65);
t2=t2[:-1]
t3=np.linspace(np.pi,3*np.pi,65);
t3=t3[:-1]
y = np.sin(np.sqrt(2)*t1)
fig,ax = plt.subplots(figsize=(7,6))
plt.plot(t1,y,'bo',lw=2)
plt.plot(t2,y,'ro',lw=2)
plt.plot(t3,y,'ro',lw=2)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$")
plt.grid()

## 1D : Bode Plot of ramp signal
t = np.linspace(-np.pi,np.pi,65)
t = t[:-1]
dt = t[1] - t[0]
fmax = 1/dt
w = np.linspace(-np.pi*fmax,np.pi*fmax,65)
w = w[:-1]
y = t
y[0] = 0
y = np.fft.fftshift(y)
Y = np.fft.fftshift(np.fft.fft(y))/64.0
fig ,ax = plt.subplots(figsize=(7,7))
plt.semilogx(np.abs(w),20*np.log10(np.abs(Y)))
plt.xlim([1,10])
plt.ylim([-20,0])
plt.xticks([1,2,3,4,6,10],["1","2","3","4",'6','10'],size=16)
plt.xlabel(r"$\omega \longrightarrow$",size = 16)
plt.ylabel(r"$|Y|$ in dB$\longrightarrow$",size = 16)
plt.title(r"Bode magnitude plot of Ramp")
plt.grid()

## 1E : Plotting the window hammed version of sin(root2*t)
t1=np.linspace(-np.pi,np.pi,65);
t1=t1[:-1]
t2=np.linspace(-3*np.pi,-np.pi,65);
t2=t2[:-1]
t3=np.linspace(np.pi,3*np.pi,65);
t3=t3[:-1]
n=np.arange(64)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/63))
y = np.sin(np.sqrt(2)*t1)*wnd
fig,ax = plt.subplots(figsize=(7,6))
plt.plot(t1,y,'bo',lw=2)
plt.plot(t2,y,'ro',lw=2)
plt.plot(t3,y,'ro',lw=2)
plt.ylabel(r"$y$",size=16)
plt.xlabel(r"$t$",size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$")
plt.grid()

## 1F : Spectrum of window hammed version
window_dft(-np.pi,np.pi,64,sinroot2,8,r'$\sin\left(\sqrt{2} t\right)$')

## 1G : Increasing the window size and samples
window_dft(-4*np.pi,4*np.pi,256,sinroot2,8,r'$\sin\left(\sqrt{2} t\right)$')

## 1H : Spectrum of window-hammed sin(1.5t)
window_dft(-4*np.pi,4*np.pi,512,sina,8,r'$sin(1.5t)$')

# Question 2a: Spectrum of cos^3(0,86t)
dft(-4*np.pi,4*np.pi,256,cos3,4,r'$cos^3(0.86t)$')

# Question 2b: Spectrum of window hammed version
window_dft(-4*np.pi,4*np.pi,256,cos3,4,r'$cos^3(0.86t)$')

# Question 3 : Estimating w and delta
## We already defined the helper functions above
print("Estimation of w and delta for given signal")
estimation_test(y_generator,1.3,-np.pi/2,25)

# Question 4 : Estimating w and delta with whitenoise
print("Estimation of w and delta for given signal with white noise")
estimation_test(y_whitenoise,1.3,-np.pi/2,25)

# Question 5 : 
## Spectrum of signal without hamming window
dft(-np.pi,np.pi,1024,chirp,70,'Chirp')
## Spectrum of signal with hamming window
window_dft(-np.pi,np.pi,1024,chirp,75,r'Chirp',antisymmetry = False)

# Question 6: Surface plots

size = 64
batches = 1024//64
t = np.linspace(-np.pi,np.pi,1025)
t= t[:-1]
fmax = 1/(t[1] - t[0])
w = np.linspace(-np.pi*fmax,np.pi*fmax,65)
w = w[:-1]
y = chirp(t)
n = np.arange(1024)
wnd=np.fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/(1023)))
y_hw = y*wnd
t_batch = np.split(t, batches)
y_batch = np.split(y, batches)
y_hw_batch = np.split(y_hw, batches)
Y = np.zeros((batches,size),dtype=complex)
Y_hw = Y.copy()
for i in range(batches):
    Y[i] = np.fft.fftshift(np.fft.fft(y_batch[i]))/size
    Y_hw[i] = np.fft.fftshift(np.fft.fft(y_hw_batch[i]))/size
tm = t[::64]
T, W = np.meshgrid(tm, w)

## Surface plot of non window-hammed signal
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(211, projection='3d')   # open a new figure
plt.title(r'Magnitude of $|Y|$')
plt.xlabel(r'+ $\omega\longrightarrow$')
plt.ylabel(r'+ $t\longrightarrow$')
surf = ax.plot_surface(W,T, np.abs(Y).T,cmap=plt.cm.jet)
fig.colorbar(surf)
ax = fig.add_subplot(212, projection='3d')
plt.title(r'$\phi$ of $|Y|$')
plt.xlabel(r'+ $\omega\longrightarrow$')
plt.ylabel(r'+ $t\longrightarrow$')
surf = ax.plot_surface(W,T, np.angle(Y).T,cstride =1 ,cmap=plt.cm.jet)
fig.colorbar(surf)

## Surface plot of window hammed signal
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(211, projection='3d')   # open a new figure
plt.title(r'Magnitude of $|Y|$')
plt.xlabel(r'+ $\omega\longrightarrow$')
plt.ylabel(r'+ $t\longrightarrow$')
surf = ax.plot_surface(W,T, np.abs(Y_hw).T,cmap=plt.cm.jet)
fig.colorbar(surf)
ax = fig.add_subplot(212, projection='3d')
plt.title(r'$\phi$ of $|Y|$')
plt.xlabel(r'+ $\omega\longrightarrow$')
plt.ylabel(r'+ $t\longrightarrow$')
surf = ax.plot_surface(W,T, np.angle(Y_hw).T,cstride =1 ,cmap=plt.cm.jet)
fig.colorbar(surf)

plt.show()