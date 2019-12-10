#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is an extension of the line fault code written by @author:amt.

This codes solves for pressure evolution coupled with velocity and State evolution in dimensional form.

The equations of pressure, state and velocity evolution are taken from Segall et al., 2010 JGR. 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class set_parameters:

# This class sets the parameters that are required to solve the coupled differential equations of line faults.

	def __init__(self):	
		self.G=30e9 # rigidity (Pa)
		self.nu=0.25  # Poisson Ratio
		self.vpl=1e-8 # plate velocity (m/s)
		self.vc= 100.0       #  40*self.vpl # cutoff velocity (m/s)
		self.a=0.001 # frictional parameter a
		self.b=0.01 # frictional parameter b
		self.dc= 0.04     # 1e-3 # weakening distance (m)
		self.vs=3e3 # shear wave velocity (m/s)
		self.sl=1 # slip law or ageing law
		self.W=60000 # asperity size (m)    
		self.WpoW=5 # this is W'/W (must be odd)
		self.gridfac=8 # this is what you add to either side to make the FFT from -inf to inf
		self.gridlength=(2*self.gridfac+self.WpoW)*self.W # simulation space (m) 
		self.sigma=10000000; # effective normal stress in Pa
		self.Lb=self.G*self.dc/(self.sigma*self.b); # L_b
		self.hstar=self.b/(self.b-self.a)*self.Lb # L_(b-a)
		self.Linf=1/np.pi*(self.b/(self.b-self.a))**2*self.Lb
		self.eta=self.G/(2*self.vs) # radiation damping (Pa-s/m)  
		if self.sl==1:
		    self.dx=self.Lb/20; # cell size (m) for slip law
		else:
		    self.dx=self.Lb/10; # cell size (m) for ageing law
		if np.mod(self.W/self.dx,1)!=0:
		    tmp=np.ceil(self.W/self.dx);
		    self.dx=self.W/tmp;
		self.WN=int(self.W/self.dx);
		self.N=int(self.gridlength/self.dx); # number of cells         
		self.X=np.arange(-(self.gridfac+np.floor(self.WpoW/2))*self.W, ((self.gridfac+(self.WpoW-np.floor(self.WpoW/2)))*self.W), self.dx).T    
		self.Wstart=np.where(self.X==0)[0]  # index of start of fault position (i.e. 0)
		if len(self.Wstart)==0:
		    self.Wstart=int(np.where(abs(self.X)==np.min(abs(self.X)))) # this deals with irrational o.dx
		else:
		    self.Wstart=int(self.Wstart[0])
		self.Wend=self.Wstart+int(self.W/self.dx)-1 # end of fault position (i.e.0) 
		self.Fstart=int(self.Wstart-np.floor(self.WpoW/2)*self.WN)
		self.Fend=int(self.Wend+np.floor(self.WpoW/2)*self.WN)   
		self.k=abs(2*np.pi*np.arange(-self.N/2,self.N/2))/((self.gridlength-self.dx)) # wavenumber vector
		self.aoverb=self.b/self.a*np.ones((self.N)) 
		self.aoverb[self.Wstart:self.Wend] = self.a/self.b
		self.agrid=self.aoverb*self.b
		self.bgrid=self.a/self.aoverb
		

		self.f_o= 0.6   # Reference rate and state func
		self.v_o= 1e-5;  # reference velocity (m/sec) 
		# pressure related parameters
		self.pressure_inf = 100000.   #  Pressure outside fault zone in Pa;
		self.pressure_initial = 10000. # Initial Pressure within fault zone in Pa;
		self.t_f = 40.e2
		self.epsilon= 1.e-5
		self.beta= 8.e-11  # 1/Pa units;
		self.sigma_minus_p_inf = self.sigma - self.pressure_inf

		self.plots= 1 	
		if self.plots:
			fig, axs = plt.subplots(2, sharex=True)       
			axs[0].plot(self.X,self.aoverb,color='black', linestyle='--')
			axs[0].axvline(x=self.X[self.Wstart],color='blue')
			axs[0].text(self.X[self.Wstart],1,'Wstart',rotation=90,color='blue')
			axs[0].axvline(x=self.X[self.Wend],color='blue')
			axs[0].text(self.X[self.Wend],1,'Wend',rotation=90,color='blue')
			axs[0].axvline(x=self.X[self.Fstart],color='red')
			axs[0].text(self.X[self.Fstart],1,'Fstart',rotation=90,color='red')
			axs[0].axvline(x=self.X[self.Fend],color='red')
			axs[0].text(self.X[self.Fend],1,'Fend',rotation=90,color='red')
			axs[0].set_ylabel('a/b')
			axs[1].plot(self.X,np.ones(len(self.X))*self.sigma,color='black', linestyle='--')
			axs[1].axvline(x=self.X[self.Wstart],color='blue')
			axs[1].axvline(x=self.X[self.Wend],color='blue')
			axs[1].axvline(x=self.X[self.Fstart],color='red')
			axs[1].axvline(x=self.X[self.Fend],color='red')
			axs[1].set_ylabel('Sigma (Pa)')

# End of the class. Now the function definitions:

def set_initial_conditions():
# This function sets the initial conditions for the coupled equations
	seed=2.;
	State = o.dc/o.vpl*np.concatenate((np.ones(o.Wstart), 2*np.ones(o.Wend-o.Wstart), np.ones(o.N-o.Wend))) # initial state 
	Vel = o.vpl*np.concatenate((np.ones(o.Wstart), 1/seed*np.ones(o.Wend-o.Wstart), np.ones(o.N-o.Wend))) # initial velocity 
	Disp = np.zeros(np.shape(Vel)) # initial slip (m)
	Pressure = o.pressure_initial * np.concatenate((np.ones(o.Wstart), 1/seed*np.ones(o.Wend-o.Wstart), np.ones(o.N-o.Wend)))
	y0=np.concatenate((State,Vel,Disp, Pressure))

	plt.figure(figsize=(8,10))
	plt.subplot(4,1,1)
	plt.plot(o.X/o.W,State)
	plt.title('Initial conditions (s)')
	plt.ylabel('Initial State (s)')
	plt.subplot(4,1,2)
	plt.plot(o.X/o.W,Vel)
	plt.ylabel('Initial Velocity (m/s)')
	plt.subplot(4,1,3)
	plt.plot(o.X/o.W,Disp)
	plt.ylabel('Initial displacement (m)')
	plt.subplot(4,1,4)
	plt.plot(o.X/o.W,Pressure)	
	plt.ylabel('Initial Pressure (Pa)')
	plt.xlabel('X/W')
	plt.show()
	return y0



def getstress(v,o):
# This function calculates the fourier term
    fftx = np.fft.fftshift(np.fft.fft(v))
    df = np.fft.ifft(np.fft.ifftshift(o.k*fftx))
    return df

def rat_stat_func(vel, state, o):
# This function calculates the rate-state function
    rsf = o.f_o + o.agrid * np.log(vel/o.v_o) + o.bgrid * np.log(state * o.v_o/o.dc) 
    return rsf


# ODEs to solve
def ratestate_sl(t,y,o):
    """
    Defines the differential equations for the coupled spring-mass system.
    Arguments:
        t :  time
        y :  vector of the state variables [theta,velocity,slip,]
        o :  structure with parameters
    """
    State=y[:o.N]
    Vel=y[o.N:2*o.N] 
    Disp=y[2*o.N:3*o.N]
    Pressure=y[3*o.N:4*o.N]
    
    # Slip law
    # State evoution
    dStatedt =  np.concatenate((np.zeros(o.Fstart),
        -State[o.Fstart:o.Fend]*Vel[o.Fstart:o.Fend]/o.dc*np.log(State[o.Fstart:o.Fend]*Vel[o.Fstart:o.Fend]/o.dc),
        np.zeros(o.N-o.Fend)))
    
    p_inf_minus_p = o.pressure_inf - Pressure
    dPressuredt =  np.concatenate((np.zeros(o.Fstart),
        + o.epsilon * dStatedt[o.Fstart:o.Fend]/ (State[o.Fstart:o.Fend] * o.beta) + p_inf_minus_p[o.Fstart:o.Fend]/o.t_f,
        np.zeros(o.N-o.Fend)))

    # Vel evoution
    stress= getstress(Vel,o);
    rsf = rat_stat_func(Vel, State, o)
    sigma_minus_p = o.sigma - Pressure
    dVeldt = np.concatenate((np.zeros(o.Fstart),
        (o.agrid[o.Fstart:o.Fend] * sigma_minus_p[o.Fstart:o.Fend]/Vel[o.Fstart:o.Fend] + o.eta)**(-1)*
        (-o.bgrid[o.Fstart:o.Fend]*dStatedt[o.Fstart:o.Fend] * sigma_minus_p[o.Fstart:o.Fend]/State[o.Fstart:o.Fend]\
        						 - o.G *stress[o.Fstart:o.Fend]/2. + rsf[o.Fstart:o.Fend]* dPressuredt[o.Fstart:o.Fend]),
        np.zeros(o.N-o.Fend)))
    
    dDispdt = Vel
    dy = np.concatenate((dStatedt,dVeldt,dDispdt,dPressuredt ))
    print(t)
    return dy


def plot_outputs(wsol):
	fig, ax = plt.subplots(figsize=(5,8))   # figure 1
	x = wsol.t[::10]
	y = o.X[o.Fstart:o.Fend]
	X, Y = np.meshgrid(x, y)
	vel = wsol.y[o.Fstart+o.N:o.Fend+o.N,::10]
	cs = ax.pcolormesh(Y, X, np.log10(vel), cmap='jet')
	cbar = fig.colorbar(cs)
	plt.ylabel('Time (s)')
	plt.xlabel('Distance (m)')
	cbar.set_label('log(V)')
	plt.show()	    

	# figure 2
	co_seis_mid = int( (o.Wend - o.Wstart)/2. )       # This is the grid point at the center of asperity
	plt.plot(wsol.t[::10] , wsol.y[o.Wstart + o.N + co_seis_mid , ::10] )
	plt.ylabel('Slip rate (m/sec)')
	plt.xlabel('Time (s)')
	plt.show()



# ------------------------------------------------------------------------------------------------------
# main program
# ------------------------------------------------------------------------------------------------------


## Set parameters
o = set_parameters()

## Initialize model
y0 = set_initial_conditions()

# Call the ODE solver
wsol = solve_ivp(lambda t, y: ratestate_sl(t, y, o), [0,5e8], y0, max_step=1000000, rtol=10e-10, atol=10e-10)

#plot outputs
plot_outputs(wsol)



