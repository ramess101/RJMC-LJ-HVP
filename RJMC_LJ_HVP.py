"""
    This code performs Reversible Jump Markov Chain Monte Carlo for the 
    Lennard-Jones fluid. The target property is heat of vaporization, which
    only depends on epsilon. Therefore, the expected outcome is that RJMC 
    favors the single parameter model (just epsilon) over the two parameter
    model (both epsilon and sigma).
    
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from LJ_fluid_correlations import *
from scipy.stats import distributions

# Here we have chosen argon as the test case
compound="argon"
fname = compound+".yaml"

# Load property values for compound

with open(fname) as yfile:
    yfile = yaml.load(yfile)

eps_lit = yfile["force_field_params"]["eps_lit"] #[K]
sig_lit = yfile["force_field_params"]["sig_lit"] #[nm]
T_c_RP = yfile["physical_constants"]["T_c"] #[K]
rho_c_RP = yfile["physical_constants"]["rho_c"] #[kg/m3]
M_w = yfile["physical_constants"]["M_w"] #[gm/mol]

# Initial guesses for epsilon and sigma are obtained from the critical constants
eps_T_c = T_c_RP / T_c_star #[K]
sig_rho_c = (rho_c_star / rho_c_RP *  M_w  / N_A * m3_to_nm3 * gm_to_kg)**(1./3) #[nm]

data = np.array(pd.read_csv('HVP_data_argon.csv'))
T = data[:,0] #[K]
RP_HVP = data[:,1] #[kJ/mol]

# Specify property and LJ correlation model used in analysis (HVP)
prop_fit = RP_HVP
prop_fit_hat = lambda T, eps, sig: HVP_hat_LJ(T,eps)
data_t = 1 # Precision in data (1/SD**2)

# Initial values for the Markov Chain
guess = (eps_T_c, sig_rho_c, 1) # Epsilon, Sigma, Precision
# Initial estimates for standard deviation used in proposed distributions of MCMC
guess_var = [1, 0.1, 0.2]

# Simplify notation
dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf

rnorm = np.random.normal
runif = np.random.rand

def calc_posterior(eps, sig, t):

    logp = 0
    #Priors on eps, sig
    logp += dnorm(sig, guess[1], 0.01)
    logp += dnorm(eps, guess[0], 1) 
    # Prior on t (precision)
    logp += dgamma(t, 0.01, 0.01)
    # Calculate property value for given eps, sig
    prop_hat = prop_fit_hat(T,eps,sig) # With HVP
    # Data likelihood
    logp += sum(dnorm(prop_fit, prop_hat, t**-2)) # With HVP
    
    return logp

def RJMC_tuned(calc_posterior,n_iterations, initial_values, prop_var, 
                     tune_for=None, tune_interval=100):
    
    n_params = len(initial_values)
            
    # Initial proposal standard deviations
    prop_sd = prop_var
    
    # Initialize trace for parameters
    trace = np.empty((n_iterations+1, n_params))
    
    # Set initial values
    trace[0] = initial_values

    # Initialize acceptance counts
    accepted = [0]*n_params
               
    # Initialize trace for model
    model_params = np.empty((n_iterations+1, 1))
    
    # Set initial values
    model_params[0] = 1
    model_change = 0
    sigma_change = 0
    
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    if tune_for is None:
        tune_for = n_iterations/2
    
    for i in range(n_iterations):
    
        if not i%1000: print('Iteration '+str(i))
    
        # Grab current parameter values
        current_params = trace[i].copy()
        current_model = model_params[i].copy()
    
        for j in range(n_params):
    
            # Get current value for parameter j
            p = current_params.copy() # This approach updates previous p values
            
            # Propose new value
            if j == 1:
                if np.random.random() < 0.5: #Use models with equal probability
                    theta = current_params[j] # Does not change sigma
                    proposed_model = 1
                else:
                    proposed_model = 2
                    theta = rnorm(current_params[j], prop_sd[j])
                    sigma_change = sigma_change + 1 #Keep track of how many times sigma actually changes
            else:
                theta = rnorm(current_params[j], prop_sd[j])
                                                    
            # Insert new value 
            p[j] = theta
    
            # Calculate log posterior with proposed value
            proposed_log_prob = calc_posterior(*p)
    
            # Log-acceptance rate (all other terms in RJMC are 1 in this case)
            alpha = proposed_log_prob - current_log_prob
    
            # Sample a uniform random variate
            u = runif()
    
            # Test proposed value
            if np.log(u) < alpha:
                # Accept
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
                current_params[j] = theta
                if j == 1:
                    model_params[i+1] = proposed_model #Keep track of which model is used for each step
                    if i > tune_for:
                        model_change += abs(current_model-proposed_model) #Keep track of how often model changes during production
                    if proposed_model == 2:
                        accepted[j] += 1 #Count accepted sigma changes only if sigma actually was a parameter
                else:
                    accepted[j] += 1
            else:
                # Reject
                trace[i+1,j] = trace[i,j]
                if j == 1:
                    model_params[i+1] = current_model
            
            # Tune every 100 iterations
            if (not (i+1) % tune_interval) and (i < tune_for):

                if j == 1:
                    acceptance_rate = (1.*accepted[j])/sigma_change # I am only not counting when the 1 parameter model goes to the 1 parameter model               
                    sigma_change = 0
                else:
                    acceptance_rate = (1.*accepted[j])/tune_interval             
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.9
                elif acceptance_rate>0.5:
                    prop_sd[j] *= 1.1                  

                accepted[j] = 0              
                        
    accept_prod = np.array([accepted[0]/(n_iterations - tune_for),accepted[1]/sigma_change,accepted[2]/(n_iterations - tune_for)])
                
    return trace, trace[tune_for:], accept_prod, model_change, model_params

n_iter = 20000
tune_for = 9000
trace_all,trace_tuned, acc_tuned, model_swaps, model_params = RJMC_tuned(calc_posterior, n_iter, guess, prop_var=guess_var, tune_for=tune_for)

model_count = np.array([np.count_nonzero(model_params[tune_for:]-1),np.count_nonzero(model_params[tune_for:]-2)])

print('Acceptance Rate during production for eps, sig, t: '+str(acc_tuned))

print('Acceptance model swap during production: '+str(model_swaps/(n_iter-tune_for)))

p_1 = 1.*model_count[1]/(n_iter-tune_for)
print('Percent that 1-parameter model is sampled: '+str(p_1 * 100.)) #The percent that use 1 parameter model

BF_1 = 1./(1./p_1 - 1)
print('Bayes Factor for 1-parameter model: '+str(BF_1)) # A value greater than 10 is strong evidence
   
f, axes = plt.subplots(3, 2, figsize=(6,6))     
for param, samples, samples_tuned, iparam in zip(['$\epsilon (K)$', '$\sigma (nm)$', 'precision'], trace_all.T,trace_tuned.T, [0,1,2]):
    axes[iparam,0].plot(samples)
    axes[iparam,0].set_ylabel(param)
    axes[iparam,0].set_xlabel('Iteration')
    axes[iparam,1].hist(samples_tuned)
    axes[iparam,1].set_xlabel(param)
    axes[iparam,1].set_ylabel('Count')
    
plt.tight_layout(pad=0.2)

f.savefig(compound+"_Trace_RJMC.pdf")

f, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].scatter(np.arange(0,n_iter+1),model_params,s=0.01)
axes[0].set_ylabel('Number of Parameters')
axes[1].hist(model_params[tune_for:n_iter+1])

plt.tight_layout(pad=0.2)

f.savefig(compound+"_Model_Params_RJMC.pdf")

#f = plt.figure()
#plt.scatter(trace_tuned[:,1],trace_tuned[:,0],label='Bayesian')
#plt.scatter(sig_lit,eps_lit,label='Literature')
#plt.scatter(sig_rho_c,eps_T_c,label='Critical Point')
#plt.scatter(guess[1],guess[0],label='Guess')
#plt.xlabel('$\sigma (nm)$')
#plt.ylabel('$\epsilon (K)$')
#plt.legend()
#
#f.savefig(compound+"_Param_RJMC.pdf")  

f = plt.figure()
plt.scatter(trace_all[:,1],trace_all[:,0],label='Trajectory')
plt.scatter(trace_tuned[:,1],trace_tuned[:,0],label='Production')
plt.scatter(sig_lit,eps_lit,label='Literature')
plt.scatter(sig_rho_c,eps_T_c,label='Critical Point')
plt.scatter(guess[1],guess[0],label='Guess')
plt.xlabel('$\sigma (nm)$')
plt.ylabel('$\epsilon (K)$')
plt.legend()

f.savefig(compound+"_Trajectory_RJMC.pdf")  

T_plot = np.linspace(T.min(), T.max())
   
f = plt.figure()

plt.plot(T,RP_HVP,'k--',label='RefProp')

for i in range(100): #Plot 100 random samples from production
    eps_sample, sig_sample, t_sample = trace_tuned[np.random.randint(0, n_iter - tune_for)]
    HVP_sample = HVP_hat_LJ(T_plot,eps_sample)
    
    plt.plot(T_plot,HVP_sample,'r',label='LJ')
    plt.xlabel("$T$ (K)")
    plt.ylabel(r"$\Delta H_v \left(\frac{kJ}{mol}\right)$")
           
# I have this redundant so that the RefProp curves are on top
# I use legend because the loop creates a lot of labels.
plt.plot(T,RP_HVP,'k--',label='RefProp')
plt.legend(['RefProp','LJ'])

plt.tight_layout(pad=0.2)

f.savefig(compound+"_Prop_RJMC.pdf")
