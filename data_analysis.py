import numpy as np
import pandas as pd
import scipy.stats
import csv
from scipy.stats import multinomial
from scipy.stats import poisson
from scipy.optimize import minimize
import math
import random
import matplotlib.pyplot as plt

# Observed counts of exposed and unexposed individuals (x are the exposed)
x111, x110, x101, x100, x011, x010, x001 = 189, 18, 47, 44, 128, 35, 47
y111, y110, y101, y100, y011, y010, y001 = 155, 16, 35, 31, 110, 33, 33
Nx, Ny = 508, 413

def neg_loglikelihood(p):
    t = p[0]
    a1 = p[1]
    a2 = p[2]
    a3 = p[3]
    a12 = p[4]
    a13 = p[5]
    a23 = p[6]
    gx = p[7]
    gy = p[8]
    p111 = np.exp(a1+a2+a3+a12+a13+a23+3*t)/((1+np.exp(a1+t))*(1+np.exp(a2+a12+t))*(1+np.exp(a3+a13+a23+t)))
    p110 = np.exp(a1+a2+a12+2*t)/((1+np.exp(a1+t))*(1+np.exp(a2+a12+t))*(1+np.exp(a3+a13+a23+t)))
    p101 = np.exp(a1+a3+a13+2*t)/((1+np.exp(a1+t))*(1+np.exp(a2+a12+t))*(1+np.exp(a3+a13+t)))
    p011 = np.exp(a2+a3+a23+2*t)/((1+np.exp(a1+t))*(1+np.exp(a2+t))*(1+np.exp(a3+a23+t)))
    p100 = np.exp(a1+t)/((1+np.exp(a1+t))*(1+np.exp(a2+a12+t))*(1+np.exp(a3+a13+t)))
    p010 = np.exp(a2+t)/((1+np.exp(a1+t))*(1+np.exp(a2+t))*(1+np.exp(a3+a23+t)))
    p001 = np.exp(a3+t)/((1+np.exp(a1+t))*(1+np.exp(a2+t))*(1+np.exp(a3+t)))
    p000 = 1/((1+np.exp(a1+t))*(1+np.exp(a2+t))*(1+np.exp(a3+t)))
    q111 = np.exp(a1+a2+a3+a12+a13+a23)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13+a23)))
    q110 = np.exp(a1+a2+a12)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13+a23)))
    q101 = np.exp(a1+a3+a13)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13)))
    q011 = np.exp(a2+a3+a23)/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3+a23)))
    q100 = np.exp(a1)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13)))
    q010 = np.exp(a2)/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3+a23)))
    q001 = np.exp(a3)/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3)))
    q000 = 1/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3)))
  
    l = (x111*np.log(gx*p111) + x110*np.log(gx*p110) + x101*np.log(gx*p101) + x011*np.log(gx*p011) + 
         x100*np.log(gx*p100) + x010*np.log(gx*p010) + x001*np.log(gx*p001) + gx*(p000 - 1) +
         y111*np.log(gy*q111) + y110*np.log(gy*q110) + y101*np.log(gy*q101) + y011*np.log(gy*q011) + 
         y100*np.log(gy*q100) + y010*np.log(gy*q010) + y001*np.log(gy*q001) + gy*(q000 - 1))

    return -l
    
# Optimize the negative log-likelihood starting from B different points   
B = 10
best_lik = 0
for b in range(B):
    init = [np.random.normal() for i in range(7)] + [random.randint(508,600)] + [random.randint(413,500)]
    res = minimize(neg_loglikelihood, init, method='nelder-mead')
    if neg_loglikelihood(res.x) < best_lik:
        best_lik = neg_loglikelihood(res.x)
        best_par = res.x
    print(b, neg_loglikelihood(init), neg_loglikelihood(res.x))
    
# Print the best estimate
best_par







