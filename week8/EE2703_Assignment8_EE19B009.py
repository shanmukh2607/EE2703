"""
This code is written by BACHOTTI SAI KRISHNA SHANMUKH EE19B009
    EE2703 Assignment 7 Solution
    The Laplace Transform
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy
import scipy.signal as sp
import warnings
warnings.filterwarnings('ignore')

sympy.init_session

def lowpass(R1,R2,C1,C2,G,Vi):
    """
    This function takes the circuit parameters of Butterworth Low Pass filter as input
    Returns symbolic node voltage solution vector, matrix and source vectors.
    """
    s = sympy.symbols('s')
    A = sympy.Matrix([[0, 0, 1, -1/G],[-1/(1+s*R2*C2), 1, 0, 0],[0, -G, G, 1],[-1/R1 -1/R2 -s*C1, 1/R2, 0 ,s*C1]])
    b = sympy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return A,b,V

def highpass(R1,R3,C1,C2,G,Vi):
    """
    This function takes the circuit parameters of High Pass filter as input
    Returns symbolic node voltage solution vector, matrix and source vectors.
    """
    s = sympy.symbols('s')
    A = sympy.Matrix([[0,0,1,-1/G],[0,-G,G,1],[-s*C2, 1/R3+s*C2, 0, 0],[-s*C1-s*C2-1/R1, s*C2, 0, 1/R1]])
    b = sympy.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return A,b,V

def sym2sig(SymX):
    """
    This function takes symbolic expression of Transfer function and
    returns numerator and denominator polynomial coefficients
    """
    s = sympy.symbols('s')
    X = sympy.simplify(SymX)
    n,d = sympy.fraction(X)
    n,d = sympy.Poly(n,s), sympy.Poly(d,s)
    num,den = n.all_coeffs(), d.all_coeffs()
    num,den = [float(f) for f in num], [float(f) for f in den]
    return num,den

def stepresponse(SymX,tstop):
    """
    This function makes use of sym2sig function to find the Transfer function of the system
    SymX : Symbolic Expression
    tstop : Stop time
    Returns the step response
    """
    num,den = sym2sig(SymX)
    den.append(0)    # Equivalent to multiplying s in the denominator polynomial
    H1 = sp.lti(num,den)
    return sp.impulse(H1,None,np.linspace(0,tstop,100000))

def input_response(SymX,f,tstop):
    """
    This function makes use of sym2sig function to find the Transfer function of the system
    SymX : Symbolic Expression
    f : Input signal as a function
    tstop : Stop time
    Returns the system response for given input
    """
    num,den = sym2sig(SymX)
    H = sp.lti(num,den)
    t = np.linspace(0,tstop,100000)
    t,y,temp = sp.lsim(H,f(t),t)
    return t,y

def sinusoids(t):
    """
    This function has two sinusoids in 1kHz and 1MHz frequencies
    """
    return np.sin(2000*np.pi*t)+np.cos(2*1e6*np.pi*t)

def damped_sinusoid(w,decay):
    """
    A function that for exponentially decaying sinusoid
    w: Frequency (in Hz)
    decay : Decay factor 
    Returns the lambda function
    """
    f = lambda t : (np.exp(-decay*t)*np.cos(2*np.pi*w*t))
    return f

def bode_plotter(vo, title,plt_num):
    """
    This function takes symbolic expression as input and outputs bode plots
    using matplotlib.pyplot
    """
    s = sympy.symbols('s')
    w = np.logspace(0,8,801)
    ss = 1j*w
    hf = sympy.lambdify(s,vo,'numpy')
    v_val = hf(ss)
    fig,ax = plt.subplots(num=plt_num,figsize=(8,7))
    ax.loglog(w,np.abs(v_val),lw=2)
    plt.xlabel(r' $\omega\longrightarrow$')
    plt.ylabel(r' Magnitude $\longrightarrow$')
    plt.title('Magnitude response of '+title+' Filter')
    plt.grid()
    fig,ax = plt.subplots(num= plt_num+1,figsize=(8,7))
    ax.semilogx(w,np.degrees(np.angle(v_val)),lw=2)
    plt.xlabel(r' $\omega\longrightarrow$')
    plt.ylabel(r' $\phi\longrightarrow$')
    plt.title('Phase response of '+title+' Filter')
    plt.grid()
    
def signal_plotter(t,y,plt_num,title_text):
    """
    A simple function for plotting a signal
    Easy to call the function that repeated usage of same block
    """
    fig,ax = plt.subplots(num=plt_num,figsize=(7,6))
    ax.plot(t,y,'r')
    plt.xlabel(r' $t\longrightarrow$')
    plt.ylabel(r' $vo\longrightarrow$')
    plt.title(title_text)
    plt.grid()

# Low Pass Filter Bode plots    
A,b,v_lpf = lowpass(10000,10000,1e-9,1e-9,1.586,1)
vo_lpf = v_lpf[3]
bode_plotter(vo_lpf,'Low Pass Butterworth',1)

# Step Response for LPF
t1, y1 = stepresponse(vo_lpf,1e-3)
signal_plotter(t1,y1,3,'Step Response of Butterworth Low Pass Filter')

# Sinusoids Input response for LPF
t2,y2 = input_response(vo_lpf,sinusoids,1e-2)
signal_plotter(t2,y2,4,'Response of Low Pass Filter for 1kHz + 1MHz sinusoids')

# High Pass Filter Bode plots
A,b,v_hpf = highpass(10000,10000,1e-9,1e-9,1.586,1)
vo_hpf = v_hpf[3]
bode_plotter(vo_hpf,'High Pass',5)

# Input response for HPF : Type 1 signal 
t3, y3 = input_response(vo_hpf,damped_sinusoid(1e3,5),1e-2)
signal_plotter(t3,y3,7,r'Response of High Pass Filter for $v_i(t)$ = $exp(-5t)$cos(2000$\pi$t)')

# Input Response for HPF : Type 2 signal
t31, y31 = input_response(vo_hpf,damped_sinusoid(1e6,5000),1e-4)
signal_plotter(t31,y31,8,r'Response of High Pass Filter for $v_i(t)$ = $exp(-5000t)$cos($2x10^6\pi$t)')

# Sinusoids Input Response for HPF
t4, y4 = input_response(vo_hpf,sinusoids,1e-5)
signal_plotter(t4,y4,9,'Response of High Pass Filter for 1kHz + 1MHz sinusoids')

# Step Response for HPF
t5, y5 = stepresponse(vo_hpf,1e-3)
signal_plotter(t5,y5,10,'Step Response of High Pass Filter')

plt.show()