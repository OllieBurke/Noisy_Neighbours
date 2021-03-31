#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 08:55:10 2020

@author: aantonelli


LISA utils
"""


import numpy as np


"""
Define the LISA response function -- IMPORTANT: Doppler Shift missing here.
"""

def d_plus(alpha,theta,phi,lam):
    
    sqrt3_64 = np.sqrt(3)/64  # 
    A = -36 * np.sin(theta)**2 * np.sin(2 * alpha - 2*lam)
    B = (3 + np.cos(2*theta))*( np.cos(2*phi)*(9 * np.sin(2*lam) - np.sin(4*alpha - 2*lam))  + \
                               np.sin(2*phi)*(np.cos(4*alpha - 2*lam) - 9*np.cos(2*lam)))
    C = -4*np.sqrt(3)*np.sin(2*theta)*(np.sin(3*alpha - 2*lam - phi) - 3*np.sin(alpha - 2*lam + phi))
    return sqrt3_64 * (A + B + C)

def d_cross(alpha,theta,phi,lam):
    A = np.sqrt(3)*np.cos(theta)*(9*np.cos(2*lam - 2*phi) - np.cos(4*alpha - 2*lam - 2*phi))
    B = -6*np.sin(theta)*(np.cos(3*alpha - 2*lam - 2*phi) +3*np.cos(alpha - 2*lam + phi))
    return (A + B)/16

def F_plus(theta,phi,psi,lam,alpha):
    return 0.5*(np.cos(2*psi)*d_plus(alpha,theta,phi,lam) - np.sin(2*psi)*d_cross(alpha,theta,phi,lam)) 

def F_cross(theta,phi,psi,lam,alpha):
    return 0.5*(np.sin(2*psi)*d_plus(alpha,theta,phi,lam) + np.cos(2*psi)*d_cross(alpha,theta,phi,lam)) 


"""
Define the LISA PSD
"""

def PowerSpectralDensity(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. This version of the PSD includes
    the sky-averaging position 'penalty', which takes into account the fact that, for some
    LISA sources, the wavelength of the GWs is shorter than LISA's arms.
    
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = 19.09*10**-3    
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))  
    PSD = ((10/(3*L**2))*(Poms + (4*Pacc)/((2*np.pi*f))**4)*(1 + 0.6*(f/f0)**2) + Sc) # PSD
    
    
    where_are_NaNs = np.isnan(PSD) #In case there are nans,
    PSD[where_are_NaNs] = 1e100    #set the PSD value for them to something very high and let's be done with it.
    
    return PSD


"""
Define inner product and SNR.
"""

# Derivation of this result in personal notes. 

def inner_product(FD_signal_1_fft,FD_signal_2_fft,delta_t,PSD,n_t):
    
    """ The FD signals here are the discretized FD signals. """
    
    return 4*delta_t*np.real(sum(FD_signal_1_fft * np.conj(FD_signal_2_fft) / (n_t * PSD))) #note: n_t in denom.

def SNR2(h_discrete_fft, delta_t, PSD, n_t):
    
    return inner_product(h_discrete_fft, h_discrete_fft, delta_t, PSD, n_t)


"""
Zero padding
"""

def zero_pad(data):
    
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')

