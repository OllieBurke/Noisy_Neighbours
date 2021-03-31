import numpy as np

def units():
    GM_sun = 1.3271244*1e20
    c =2.9979246*1e8
    M_sun =1.9884099*1e30
    G = 6.6743*1e-11
    pc= 3.0856776*1e16
    pi = np.pi
    Mpc = (10**6) * pc

    return GM_sun, c, M_sun, G, Mpc, pi


def htilde_GR(f,eps,params):
    
    """
    Here we calculate a TaylorF2 model up to 2PN which takes as input the following
    set of parameters: (log of chirp mass, symmetric mass ratio, beta).
    This can easily be changed in the first few lines where the parameters are loaded.
    The main reference is https://arxiv.org/pdf/gr-qc/0509116.pdf [Eqs (3.4)].
    
    Note on distance: 
    
    Notice that the effective distance contains information about the angular dependence
    of the binary. The model can thus be used for all detectors, as long as this distance
    parameter is chosen consistently. 
    
    Note on spin: 
    
    The spin parameter beta is defined in Eq.(2.3a) in [arxiv:0411129].
    Notice that this quantity is constructed in such a way to be smaller or equal
    than 9.4, and of course it ranges from 0 (no spins) to this upper value. 
    The coefficient enters the phase as in Eq.(2.2) in the same paper.
    """
    GM_sun, c, M_sun, G, Mpc, pi = units()
    t0 =1.
    phi0 =0.
    # Load the parameters
    Mchirp_true = M_sun * np.exp(params[0])
    eta_true = params[1]
    beta_true = params[2]
    Deff = params[3]
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    # PN expansion parameter (velocity).
    
    v = (pi*G*Mchirp_true*eta_true**(-3/5)/(c**3) * f)**(1/3)
    # Amplitude explicitly given in terms of units and frequency.
    # Notice that lowest PN order here is fine. Biggest contributions from phase.
    
    amplitude_1 = - (Mpc/Deff)*np.sqrt((5/(24*pi)))*(GM_sun/(c**2 *Mpc))
    amplitude_2 = (pi*GM_sun/(c**3))**(-1/6) * (Mchirp_true/M_sun)**(5/6)
    amplitude = amplitude_1*amplitude_2 * f**(-7/6)
    # Phase: add or remove PN orders here as you see fit.
    
    psi_const = 2*pi*f*t0 - 2*phi0 - pi/4
    psi1PN = (3715/756 + (55/9)*eta_true)*v**(-3)
    psi1_5PN_tails = -16*pi*v**(-2)
    psi1_5PN_spin = 4*beta_true*v**(-2)
    
    psi2PN = (15293365/508032+(27145/504)*eta_true+(3085/72)*eta_true**2)*v**(-1)
    psi25PNlog = pi*(38645/252- (65/3) *eta_true)* np.log(v)
    psi3PN = v*(11583231236531/4694215680 - (640/3) * (pi**2) -6848/21 *np.euler_gamma
              + eta_true*(-15335597827/3048192 + (2255/12) * (pi**2) - 1760/3 * theta - 12320/9 * delta)
              + (eta_true**2) *76055/1728 - (eta_true**3) * 127825/1296 - 6848/21 * np.log(4))
    psi3PNlog = - 6848/21 *v * np.log(v)
    psi35PN = pi * v**2 * (77096675./254016 + (378515./1512) *eta_true - 74045./756 * (eta_true**2)* (1-eps))
    psi_fullPN = (3/(128*eta_true))*(v**(-5)+psi1PN+psi1_5PN_tails+psi1_5PN_spin+psi2PN
                                  + psi25PNlog + psi3PN + psi3PNlog + psi35PN)
    psi = psi_const + psi_fullPN 
    return amplitude* np.exp(-1j*psi)

def htilde_GB(f,params):
    
    """
    Here we calculate a TaylorF2 model up to 2PN which takes as input the following
    set of parameters: (log of chirp mass, symmetric mass ratio, beta).
    This can easily be changed in the first few lines where the parameters are loaded.
    The main reference is https://arxiv.org/pdf/gr-qc/0509116.pdf [Eqs (3.4)].
    
    Note on distance: 
    
    Notice that the effective distance contains information about the angular dependence
    of the binary. The model can thus be used for all detectors, as long as this distance
    parameter is chosen consistently. 
    
    Note on spin: 
    
    The spin parameter beta is defined in Eq.(2.3a) in [arxiv:0411129].
    Notice that this quantity is constructed in such a way to be smaller or equal
    than 9.4, and of course it ranges from 0 (no spins) to this upper value. 
    The coefficient enters the phase as in Eq.(2.2) in the same paper.
    """
    # Units
    
    # Load the parameters
    t0 =1.
    phi0 =0.
    GM_sun, c, M_sun, G, Mpc, pi = units()

    Mchirp_true = M_sun * np.exp(params[0])
    eta_true = params[1]
    Deff = params[2]
    

    # PN expansion parameter (velocity).
    
    v = (pi*G*Mchirp_true*eta_true**(-3/5)/(c**3) * f)**(1/3)
    # Amplitude explicitly given in terms of units and frequency.
    # Notice that lowest PN order here is fine. Biggest contributions from phase.
    
    amplitude_1 = - (1/Deff)*np.sqrt((5/(24*pi)))*(GM_sun/(c**2 ))
    amplitude_2 = (pi*GM_sun/(c**3))**(-1/6) * (Mchirp_true/M_sun)**(5/6)
    amplitude = amplitude_1*amplitude_2 * f**(-7/6)
    
    new_amplitude = -np.sqrt(5*np.pi/24)*(G*Mchirp_true/(c**3))*(G*Mchirp_true/(Deff*c**2))*(f*np.pi*G*Mchirp_true/(c**3))**(-7/6)
    
    # Phase: add or remove PN orders here as you see fit.
    
    psi_const = 2*pi*f*t0 - 2*phi0 - pi/4
#     psi1PN = (3715/756 + (55/9)*eta_true)*v**(-3)
#     psi1_5PN_tails = -16*pi*v**(-2)
#     psi1_5PN_spin = 4*beta_true*v**(-2)
    
#     psi2PN = (15293365/508032+(27145/504)*eta_true+(3085/72)*eta_true**2)*v**(-1)
#     psi25PNlog = pi*(38645/252- (65/3) *eta_true)* np.log(v)
#     psi3PN = v*(11583231236531/4694215680 - (640/3) * (pi**2) -6848/21 *np.euler_gamma
#               + eta_true*(-15335597827/3048192 + (2255/12) * (pi**2) - 1760/3 * theta - 12320/9 * delta)
#               + (eta_true**2) *76055/1728 - (eta_true**3) * 127825/1296 - 6848/21 * np.log(4))
#     psi3PNlog = - 6848/21 *v * np.log(v)
#     psi35PN = pi * v**2 * (77096675./254016 + (378515./1512) *eta_true - 74045./756 * (eta_true**2)* (1-eps))
    psi_fullPN = (3/(128*eta_true))*(v**(-5) )
                                    #+psi1PN+psi1_5PN_tails+psi1_5PN_spin+psi2PN
                                  #+ psi25PNlog + psi3PN + psi3PNlog + psi35PN)
    psi = psi_const + psi_fullPN 
    return amplitude_1,amplitude_2,np.exp(-1j*psi),new_amplitude* np.exp(-1j*psi)



def htilde_AP(f,params):
    
    """
    Here we calculate a TaylorF2 model up to 2PN which takes as input the following
    set of parameters: (log of chirp mass, symmetric mass ratio, beta).
    This can easily be changed in the first few lines where the parameters are loaded.
    The main reference is https://arxiv.org/pdf/gr-qc/0509116.pdf [Eqs (3.4)].
    
    Note on distance: 
    
    Notice that the effective distance contains information about the angular dependence
    of the binary. The model can thus be used for all detectors, as long as this distance
    parameter is chosen consistently. 
    
    Note on spin: 
    
    The spin parameter beta is defined in Eq.(2.3a) in [arxiv:0411129].
    Notice that this quantity is constructed in such a way to be smaller or equal
    than 9.4, and of course it ranges from 0 (no spins) to this upper value. 
    The coefficient enters the phase as in Eq.(2.2) in the same paper.
    """
    # Units
    
#     GM_sun = 1.3271244*1e20
#     c =2.9979246*1e8
#     M_sun =1.9884099*1e30
#     G = 6.6743*1e-11
#     pc= 3.0856776*1e16
#     pi = np.pi
#     Mpc = 10**6 * pc
    
    # Load the parameters
    
    t0 =1.
    phi0 =0.    
    GM_sun, c, M_sun, G, Mpc, pi = units()

    Mchirp_true = M_sun * np.exp(params[0])
    eta_true = params[1]
    beta_true = params[2]
    Deff = params[3]
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    # PN expansion parameter (velocity).
    
    v = (pi*G*Mchirp_true*eta_true**(-3/5)/(c**3) * f)**(1/3)
    # Amplitude explicitly given in terms of units and frequency.
    # Notice that lowest PN order here is fine. Biggest contributions from phase.
    
    amplitude_1 = - (Mpc/Deff)*np.sqrt((5/(24*pi)))*(GM_sun/(c**2 *Mpc))
    amplitude_2 = (pi*GM_sun/(c**3))**(-1/6) * (Mchirp_true/M_sun)**(5/6)
    amplitude = amplitude_1*amplitude_2 * f**(-7/6)
    # Phase: add or remove PN orders here as you see fit.
    
    psi_const = 2*pi*f*t0 - 2*phi0 - pi/4
    psi1PN = (3715/756+55/9*eta_true)*v**(-3)
    psi1_5PN_tails = -16*pi*v**(-2)
    psi1_5PN_spin = 4*beta_true*v**(-2)
    psi2PN = (15293365/508032+27145/504*eta_true+3085/72*eta_true**2)*v**(-1)
    psi25PNlog = pi*(38645/252- 65/3 *eta_true)* np.log(v)
    psi3PN = v*(11583231236531/4694215680 -640/3 * pi**2 -6848/21 *np.euler_gamma
              + eta_true*(-15335597827/3048192+2255/12 * pi**2-1760/3 * theta - 12320/9 * delta)
              + eta_true**2 *76055/1728 - eta_true**3 * 127825/1296 - 6848/21 * np.log(4))
    psi3PNlog = - 6848/21 *v * np.log(v)
    psi35PN = pi * v**2 * (77096675./254016 + 378515./1512 *eta_true - 74045./756 * eta_true**2)
    psi_fullPN = 3/(128*eta_true)*(v**(-5)+psi1PN+psi1_5PN_tails+psi1_5PN_spin+psi2PN
                                  + psi25PNlog + psi3PN + psi3PNlog + psi35PN)
    psi = psi_const + psi_fullPN 
    return amplitude* np.exp(-1j*psi)

def T_chirp(fmin,M_chirp,eta):
    t0 =1.
    phi0 =0.
    GM_sun, c, M_sun, G, Mpc, pi = units()
    M_chirp *= M_sun
    
    M = M_chirp*eta**(-3/5)
    v_low = (pi*G*M_chirp*eta**(-3/5)/(c**3) * fmin)**(1/3)
    
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    gamma = np.euler_gamma
    
    pre_fact = ((5/(256*eta)) * G*M/(c**3))
    first_term = (v_low**(-8) + (743/252 + (11/3) * eta ) * (v_low **(-6)) - (32*np.pi/5)*v_low**(-5)
                +(3058673/508032 + (5429/504)*eta + (617/72)*eta**2)*v_low**(-4)
                 +(13*eta/3 - 7729/252)*np.pi*v_low**-3)
    
    second_term = (6848*gamma/105 - 10052469856691/23471078400 + 128*pi**2/3 + (
    3147553127/3048192 - 451*(pi**2)/12)*eta - (15211*eta**2)/1728 + (2555*eta**3 / 1296) +
                   (6848/105)*np.log(4*v_low))*v_low**-2
    
    third_term = ((14809/378)*eta**2 - (75703/756) * eta - 15419335/127008)*pi*v_low**-1
    return pre_fact * (first_term + second_term + third_term)

def final_frequency(M_chirp,eta):
    GM_sun, c, M_sun, G, Mpc, pi = units()
    M_tot = M_chirp*eta**(-3/5) * M_sun
    
    return (c**3)/(6*np.sqrt(6)*np.pi*G*M_tot)


def inner_prod(sig1_f,sig2_f,PSD,delta_f):
    """
    Wiener Product with constant PSD. Here we use Parseval's theorem. Note the definition of the SNR.
    """
    return (4*delta_f)  * np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD))

def numerical_derivs(freq_bin,pars):
    logMchirp_1 = pars[0];eta_1 = pars[1];beta_1 = pars[2]; Deff_1 = pars[3]
    
    
    logMchirp_delta = 1e-5
    params_1_p = [logMchirp_1 + logMchirp_delta,eta_1,beta_1,Deff_1]
    params_1_m = [logMchirp_1 - logMchirp_delta,eta_1,beta_1,Deff_1]

    deriv_log_Mchirp_1 = (htilde_AP(freq_bin,params_1_p) - htilde_AP(freq_bin,params_1_m))/(2*logMchirp_delta)


    eta_delta = 1e-6
    params_1_p = [logMchirp_1,eta_1 + eta_delta,beta_1,Deff_1]
    params_1_m = [logMchirp_1,eta_1 - eta_delta,beta_1,Deff_1]

    deriv_log_eta_1 = (htilde_AP(freq_bin,params_1_p) - htilde_AP(freq_bin,params_1_m))/(2*eta_delta)



    beta_delta = 1e-6
    params_1_p = [logMchirp_1,eta_1,beta_1 + beta_delta,Deff_1]
    params_1_m = [logMchirp_1,eta_1,beta_1 - beta_delta,Deff_1]

    deriv_log_beta_1 = (htilde_AP(freq_bin,params_1_p) - htilde_AP(freq_bin,params_1_m))/(2*beta_delta)

    diff_vec = [deriv_log_Mchirp_1,deriv_log_eta_1,deriv_log_beta_1]
    
    
    return diff_vec

def fish_matrix(diff_vec,PSD,delta_f):
    N = len(diff_vec)
    fish_mix = np.eye(N)
    for i in range(0,N):
        for j in range(0,N):
            fish_mix[i,j] = inner_prod(diff_vec[i],diff_vec[j],PSD,delta_f)
            
            

    import mpmath as mp
    mp.dps = 4000;  

    fish_mix_prec = mp.matrix(fish_mix)

    fish_mix_inv = fish_mix_prec**-1

    Cov_Matrix = np.eye(N)
    for i in range(0,N):
        for j in range(0,N):
            Cov_Matrix[i,j] = float(fish_mix_inv[i,j])

    return Cov_Matrix

# MCMC

# MCMC
"""
Created on Mon Nov 25 23:53:26 2019

@author: Ollie
"""

import numpy as np
import scipy as sp
import random as rd
import matplotlib.pyplot as plt

def llike(pdgrm, variances):
    """
    Computes log (Whittle) likelihood 
    """

    return -0.5 * sum(pdgrm / variances)

def lprior_logM_chirp(logM_chirp,logM_chirp_low, logM_chirp_high):
    """
    Prior on amplitude - uniform
    """

    if logM_chirp < logM_chirp_low or logM_chirp > logM_chirp_high:
        print('rejected logM_chirp')
        return -1e100
    else:
        return 0
    
def lprior_eta(eta,eta_low, eta_high):
    """
    Prior on amplitude - uniform
    """
    if eta < eta_low or eta > eta_high:
        print('rejected eta')

        return -1e100
    else:
        return 0
    
def lprior_beta(beta,beta_low, beta_high):
    """
    Prior on amplitude - uniform
    """

    if beta < beta_low or beta > beta_high:
        print('rejected beta')
        return -1e100
    else:
        return 0

    
def lpost_full(pdgrm, variances,
          logM_chirp_1,logM_chirp_2, logM_chirp_low,logM_chirp_high,
          eta_1, eta_2, eta_low, eta_high,
          beta_1, beta_2,beta_low, beta_high):
    '''
    Compute log posterior
    '''
    
    return(lprior_logM_chirp(logM_chirp_1,logM_chirp_low, logM_chirp_high) +
           lprior_logM_chirp(logM_chirp_2,logM_chirp_low, logM_chirp_high) +
           lprior_eta(eta_1,eta_low, eta_high) + 
           lprior_eta(eta_2,eta_low, eta_high) +
           lprior_beta(beta_1,beta_low, beta_high) +
           lprior_beta(beta_2,beta_low, beta_high) +
           + llike(pdgrm, variances))


def accept_reject(lp_prop, lp_prev):
    '''
    Compute log acceptance probability (minimum of 0 and log acceptance rate)
    Decide whether to accept (1) or reject (0)
    '''
    u = np.random.uniform(size = 1)  # U[0, 1]
    r = np.minimum(0, lp_prop - lp_prev)  # log acceptance probability
    if np.log(u) < r:
        return(1)  # Accept
    else:
        return(0)  # Reject
    
 

def accept_rate(parameter):
    '''
    Compute acceptance rate for a specific parameter
    Used to adapt the proposal variance in a MH sampler
    Input: parameter (sequence of samples of a parameter)
    '''
    rejections = 0
    for i in range(len(parameter) - 1):  # Count rejections
        rejections = rejections + (parameter[i + 1] == parameter[i])
    reject_rate = rejections / (len(parameter) - 1)  # Rejection rate
    return(1 - reject_rate)  # Return acceptance rate
    

#####
#####
    

def MCMC_full(data_f,f, true_vals,D_vec,Cov_Matrix,
                          variances,
                           logM_chirp_high,logM_chirp_low,
                           eta_high, eta_low,
                           beta_high, beta_low,
                           Ntotal, 
                           burnin, 
                           printerval = 50):

    
    np.random.seed(2) # Set the seed
    
    
    logM_chirp_1 = []   # Initialise empty vectors
    eta_1 = []
    beta_1 = []
    Deff_1 = []
    
    logM_chirp_2 = []   # Initialise empty vectors
    eta_2 = []
    beta_2 = []
    Deff_2 = []
  

    logM_chirp_1.append(true_vals[0])
    eta_1.append(true_vals[1])
    beta_1.append(true_vals[2])
    Deff_1.append(D_vec[0])
    
    logM_chirp_2.append(true_vals[3])
    eta_2.append(true_vals[4])
    beta_2.append(true_vals[5])
    Deff_2.append(D_vec[1])


    
    
    delta_f = f[1] - f[0]   # Extract sampling interval
    
    params_1 = [logM_chirp_1[0],eta_1[0],beta_1[0],Deff_1[0]]
    params_2 = [logM_chirp_2[0],eta_2[0],beta_2[0],Deff_2[0]]

    
    signal_init_f_1 = htilde_AP(f,params_1)
    signal_init_f_2 = htilde_AP(f,params_2)
    
    signal_f_init_tot = signal_init_f_1 + signal_init_f_2 

    # Compute periodogram
    pdgrm = abs(data_f - signal_f_init_tot)**2  
    print(pdgrm)
    if pdgrm[0] == 0:
        print('There will be no bias')
    else:
        print('Prepare for bias')
                                                      
    # Initial value for log posterior
    lp = []
    lp.append(lpost_full(pdgrm, variances,
                    logM_chirp_1[0], logM_chirp_2[0],logM_chirp_low, logM_chirp_high,
                    eta_1[0], eta_2[0], eta_low, eta_high,
                    beta_1[0], beta_2[0], beta_low, beta_high))
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten

    accept_reject_count = [0]   
    #####                                                  
    # Run MCMC
    #####
    for i in range(1, Ntotal):

        if i % printerval == 0:
            print("i = ", i)  # Iteration and Acceptance/Rejection ratio 
            print("acceptance_reject ratio", 100*sum(accept_reject_count)/len(accept_reject_count),'percent')
            

        ####

        #####
        # Step 1: Sample spin, a
        #####
        
        lp_prev = lp_store  # Call previous stored log posterior
        
        # Hardcoded standard deviations because I'm a twat.
        
#         logM_chirp_prop = logM_chirp[i - 1] + np.random.normal(0,1.94471368e-05)
#         eta_prop = eta[i - 1] + np.random.normal(0,6.51506233e-04)
#         beta_prop = beta[i - 1] + np.random.normal(0,6.17458158e-03)

        prev_vec = [logM_chirp_1[i - 1], eta_1[i - 1], beta_1[i - 1],
                   logM_chirp_2[i - 1], eta_2[i - 1], beta_2[i - 1]]
    
        
        
        prop_vec = np.random.multivariate_normal(prev_vec, (1/2)*Cov_Matrix)
        
#         print(prop_vec)

        logM_chirp_prop_1 = prop_vec[0]
        eta_prop_1 = prop_vec[1]
        beta_prop_1 = prop_vec[2]
        
        logM_chirp_prop_2 = prop_vec[3]
        eta_prop_2 = prop_vec[4]
        beta_prop_2 = prop_vec[5]
        
#         print(eta_prop_1,eta_prop_2,eta_prop_3,eta_prop_4)
        
        param_1_prop = [logM_chirp_prop_1, eta_prop_1, beta_prop_1, Deff_1[0]]
        param_2_prop = [logM_chirp_prop_2, eta_prop_2, beta_prop_2, Deff_2[0]]
        
        signal_prop_f_1  = htilde_AP(f,param_1_prop)  # New proposed signal
        signal_prop_f_2  = htilde_AP(f,param_2_prop)  # New proposed signal
        
        signal_prop_f_tot = signal_prop_f_1 + signal_prop_f_2
        
        pdgrm_prop = abs(data_f - signal_prop_f_tot)**2  # Compute periodigram
        
        
        # Compute log posterior
        lp_prop = lpost_full(pdgrm_prop, variances, 
                         logM_chirp_prop_1,logM_chirp_prop_2,logM_chirp_low,logM_chirp_high,
                           eta_prop_1,eta_prop_2, eta_low, eta_high,
                             beta_prop_1, beta_prop_2, beta_low, beta_high)  # Compute proposed log posterior
        

        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            logM_chirp_1.append(logM_chirp_prop_1) 
            eta_1.append(eta_prop_1)
            beta_1.append(beta_prop_1)

            logM_chirp_2.append(logM_chirp_prop_2) 
            eta_2.append(eta_prop_2)
            beta_2.append(beta_prop_2)
            
            accept_reject_count.append(1)  # Add one to counter
              
            lp_store = lp_prop  # Overwrite lp_store

        else:  # Reject
            
            logM_chirp_1.append(logM_chirp_1[i - 1]) 
            eta_1.append(eta_1[i - 1])
            beta_1.append(beta_1[i - 1])

            logM_chirp_2.append(logM_chirp_2[i - 1]) 
            eta_2.append(eta_2[i - 1])
            beta_2.append(beta_2[i - 1])
            
            accept_reject_count.append(0)  # Add 0 to counter
              
        lp.append(lp_store)  # Add log posterior value
        


        
    return logM_chirp_1,eta_1,beta_1,logM_chirp_2,eta_2,beta_2,lp

def CV_Calc(deltaH,noise_f,waveform_errors,diff_vec,Cov_Matrix,PSD,delta_f):
    N = len(diff_vec)

    deltah = noise_f + waveform_errors + deltaH


    b_vec_n = [inner_prod(diff_vec[i],noise_f,PSD,delta_f) for i in range(0,N)]
    b_vec_waveform_errors = [inner_prod(diff_vec[i],waveform_errors,PSD,delta_f) for i in range(0,N)]
    b_vec_unresolved_signals = [inner_prod(diff_vec[i],deltaH,PSD,delta_f) for i in range(0,N)]



    biases_pred_n = np.matmul(Cov_Matrix,b_vec_n)
    biases_pred_waveform_errors = np.matmul(Cov_Matrix,b_vec_waveform_errors)
    biases_pred_unresolved = np.matmul(Cov_Matrix,b_vec_unresolved_signals)


    biases_pred_total =  (biases_pred_waveform_errors + biases_pred_unresolved +
                                     biases_pred_n )
    
    return biases_pred_n,biases_pred_waveform_errors,biases_pred_unresolved,biases_pred_total

# MCMC

# MCMC
"""
Created on Mon Nov 25 23:53:26 2019

@author: Ollie
"""

import numpy as np
import scipy as sp
import random as rd
import matplotlib.pyplot as plt

    
def lpost(pdgrm, variances,
          logM_chirp_1,logM_chirp_low,logM_chirp_high,
          eta_1, eta_low, eta_high,
          beta_1, beta_low, beta_high):
    '''
    Compute log posterior
    '''
    
    return(lprior_logM_chirp(logM_chirp_1,logM_chirp_low, logM_chirp_high) +
           lprior_eta(eta_1,eta_low, eta_high) +
           lprior_beta(beta_1,beta_low, beta_high) +
            llike(pdgrm, variances))


    

def MCMC_1_sig(data_f,f, true_vals,D_vec,Cov_Matrix,
                          variances,
                           logM_chirp_high,logM_chirp_low,
                           eta_high, eta_low,
                           beta_high, beta_low,
                           Ntotal, 
                           burnin, 
                           printerval = 50):

    
    np.random.seed(2) # Set the seed
    
    
    logM_chirp_1 = []   # Initialise empty vectors
    eta_1 = []
    beta_1 = []
    Deff_1 = []
    
      

    logM_chirp_1.append(true_vals[0])
    eta_1.append(true_vals[1])
    beta_1.append(true_vals[2])
    Deff_1.append(D_vec[0])
    
    
    
    delta_f = f[1] - f[0]   # Extract sampling interval
    
    params_1 = [logM_chirp_1[0],eta_1[0],beta_1[0],Deff_1[0]]

    
    signal_init_f_1 = htilde_AP(f,params_1)

    
    signal_f_init_tot = signal_init_f_1 

    # Compute periodogram
    pdgrm = abs(data_f - signal_f_init_tot)**2  
    if pdgrm[0] == 0:
        print('There will be no bias')
    else:
        print('Prepare for bias')
                                                      
    # Initial value for log posterior
    lp = []
    lp.append(lpost(pdgrm, variances,
                    logM_chirp_1[0], logM_chirp_low, logM_chirp_high,
                    eta_1[0], eta_low, eta_high,
                    beta_1[0], beta_low, beta_high))
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten

    accept_reject_count = [0]   
    #####                                                  
    # Run MCMC
    #####
    for i in range(1, Ntotal):

        if i % printerval == 0:
            print("i = ", i)  # Iteration and Acceptance/Rejection ratio 
            print("acceptance_reject ratio", 100*sum(accept_reject_count)/len(accept_reject_count),'percent')
            

        ####

        #####
        # Step 1: Sample spin, a
        #####
        
        lp_prev = lp_store  # Call previous stored log posterior
        
        # Hardcoded standard deviations because I'm a twat.
        
#         logM_chirp_prop = logM_chirp[i - 1] + np.random.normal(0,1.94471368e-05)
#         eta_prop = eta[i - 1] + np.random.normal(0,6.51506233e-04)
#         beta_prop = beta[i - 1] + np.random.normal(0,6.17458158e-03)

        prev_vec = [logM_chirp_1[i - 1], eta_1[i - 1], beta_1[i - 1]]
    
        
        
        prop_vec = np.random.multivariate_normal(prev_vec, Cov_Matrix)
        
#         print(prop_vec)

        logM_chirp_prop_1 = prop_vec[0]
        eta_prop_1 = prop_vec[1]
        beta_prop_1 = prop_vec[2]
        

        
#         print(eta_prop_1,eta_prop_2,eta_prop_3,eta_prop_4)
        
        param_1_prop = [logM_chirp_prop_1, eta_prop_1, beta_prop_1, Deff_1[0]]

        
        signal_prop_f_1  = htilde_AP(f,param_1_prop)  # New proposed signal
        
        signal_prop_f_tot = signal_prop_f_1
        
        pdgrm_prop = abs(data_f - signal_prop_f_tot)**2  # Compute periodigram
        
        
        # Compute log posterior
        lp_prop = lpost(pdgrm_prop, variances, 
                         logM_chirp_prop_1,logM_chirp_low,logM_chirp_high,
                           eta_prop_1, eta_low, eta_high,
                             beta_prop_1, beta_low, beta_high)  # Compute proposed log posterior
        

        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            logM_chirp_1.append(logM_chirp_prop_1) 
            eta_1.append(eta_prop_1)
            beta_1.append(beta_prop_1)
        
            
            accept_reject_count.append(1)  # Add one to counter
              
            lp_store = lp_prop  # Overwrite lp_store

        else:  # Reject
            
            logM_chirp_1.append(logM_chirp_1[i - 1]) 
            eta_1.append(eta_1[i - 1])
            beta_1.append(beta_1[i - 1])
            
            accept_reject_count.append(0)  # Add 0 to counter
              
        lp.append(lp_store)  # Add log posterior value
        


        
    return logM_chirp_1,eta_1,beta_1,lp
    

