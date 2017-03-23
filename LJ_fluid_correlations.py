import numpy as np
import math
import yaml
import scipy as sp

# Conversion constants

k_B = 1.38065e-23 #[J/K]
N_A = 6.02214e23 #[1/mol]
m3_to_nm3 = 1e27
gm_to_kg = 1./1000
J_to_kJ = 1./1000
J_per_m3_to_kPA = 1./1000

with open("LJ_fluid.yaml") as yfile:
    yfile = yaml.load(yfile)

T_c_star = yfile["correlation_parameters"]["Lofti"]["T_c_star"]
rho_c_star = yfile["correlation_parameters"]["Lofti"]["rho_c_star"]
rho_L_star_params = yfile["correlation_parameters"]["Lofti"]["rho_L_star_params"]
rho_v_star_params = yfile["correlation_parameters"]["Lofti"]["rho_v_star_params"]
P_v_star_params = yfile["correlation_parameters"]["Lofti"]["P_v_star_params"]
HVP_star_params = yfile["correlation_parameters"]["Lofti"]["HVP_star_params"]

rho_L_star_params = [rho_c_star, T_c_star, rho_L_star_params[0], rho_L_star_params[1], rho_L_star_params[2]]
rho_v_star_params = [rho_c_star, T_c_star, rho_v_star_params[0], rho_v_star_params[1], rho_v_star_params[2]]

def rho_L_star_hat(T_star, b = rho_L_star_params):
    tau = np.ones(len(T_star))*b[1] - T_star # T_c_star - T_star
    rho_L_star = b[0] + b[2]*tau**(1./3) + b[3]*tau + b[4]*tau**(3./2)
    return rho_L_star

def rho_L_hat_LJ(T,eps,sig,M_w):
    T_star = T/(np.ones(len(T))*eps)
    rho_L_star = rho_L_star_hat(T_star)
    rho_L = rho_L_star *  M_w  / sig**3 / N_A * m3_to_nm3 * gm_to_kg #[kg/m3]
    return rho_L

def rho_v_star_hat(T_star, b = rho_v_star_params):
    tau = np.ones(len(T_star))*b[1] - T_star # T_c_star - T_star
    rho_v_star = b[0] + b[2]*tau**(1./3) + b[3]*tau + b[4]*tau**(3./2)
    return rho_v_star

def rho_v_hat_LJ(T,eps,sig,M_w):
    T_star = T/(np.ones(len(T))*eps)
    rho_v_star = rho_v_star_hat(T_star)
    rho_v = rho_v_star *  M_w  / sig**3 / N_A * m3_to_nm3 * gm_to_kg #[kg/m3]
    return rho_v

def P_v_star_hat(T_star, b = P_v_star_params):
    P_v_star = np.exp(b[0]*T_star + b[1]/T_star + b[2]/(T_star**4))
    return P_v_star

def P_v_hat_LJ(T,eps,sig):
    T_star = T/(np.ones(len(T))*eps)
    P_v_star = P_v_star_hat(T_star)
    P_v = P_v_star *  eps  / sig**3 * k_B * m3_to_nm3 * J_per_m3_to_kPA #[kPa]
    return P_v

def HVP_star_hat(T_star, T_c = T_c_star, b = HVP_star_params):
    tau = np.ones(len(T_star))*T_c - T_star # T_c_star - T_star
    HVP_star = b[0]*tau**(1./3) + b[1]*tau**(2./3) + b[2]*tau**(3./2)
    return HVP_star

def HVP_hat_LJ(T,eps):
    T_star = T/(np.ones(len(T))*eps)
    HVP_star = HVP_star_hat(T_star)
    HVP = HVP_star * eps * k_B * N_A * J_to_kJ #[kJ/mol]
    return HVP

def B2_hat_LJ(T,eps,sig):
    if eps == 0:
        pass
    T_star = T/(np.ones(len(T))*eps)
    B2_star = np.zeros(len(T))
    n = np.arange(0,31)
    for i,t_star in enumerate(T_star):
        addend = pow(2,(2*n+1.)/2)*pow(1./t_star,(2*n+1.)/4)*sp.special.gamma((2*n-1.)/4)/(4*sp.misc.factorial(n))
        #B2_star[i] = addend.sum() # This is the standard approach but sometimes 'nan' results
        B2_star[i] = np.nansum(addend) # This can handle even 'nan' results
    B2_star *= -2./3 * math.pi * sig**3
    B2 = B2_star * N_A / m3_to_nm3 #[m3/mol]
    return B2

def LJ_model(r,eps,sig):
    r_star = r/sig
    U = 4 * eps * (r_star**(-12) - r_star**(-6))
    return U