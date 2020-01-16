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

# Observed counts (x are the exposed)
x111, x110, x101, x100, x011, x010, x001 = 189, 18, 47, 44, 128, 35, 47
y111, y110, y101, y100, y011, y010, y001 = 155, 16, 35, 31, 110, 33, 33
Nx, Ny = 508, 413

# Negative log-likelihood under the null H0: theta = 0
def neg_loglikelihood_null(p):
    a1 = p[0]
    a2 = p[1]
    a3 = p[2]
    a12 = p[3]
    a13 = p[4]
    a23 = p[5]
    gx = p[6]
    gy = p[7]

    q111 = np.exp(a1+a2+a3+a12+a13+a23)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13+a23)))
    q110 = np.exp(a1+a2+a12)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13+a23)))
    q101 = np.exp(a1+a3+a13)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13)))
    q011 = np.exp(a2+a3+a23)/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3+a23)))
    q100 = np.exp(a1)/((1+np.exp(a1))*(1+np.exp(a2+a12))*(1+np.exp(a3+a13)))
    q010 = np.exp(a2)/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3+a23)))
    q001 = np.exp(a3)/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3)))
    q000 = 1/((1+np.exp(a1))*(1+np.exp(a2))*(1+np.exp(a3)))
    
    l = (x111*np.log(gx*q111) + x110*np.log(gx*q110) + x101*np.log(gx*q101) + x011*np.log(gx*q011) + 
         x100*np.log(gx*q100) + x010*np.log(gx*q010) + x001*np.log(gx*q001) + gx*(q000 - 1) +
         y111*np.log(gy*q111) + y110*np.log(gy*q110) + y101*np.log(gy*q101) + y011*np.log(gy*q011) + 
         y100*np.log(gy*q100) + y010*np.log(gy*q010) + y001*np.log(gy*q001) + gy*(q000 - 1))

    return -l
    
# Start the optimization in B different points
B = 10
best_lik = 0
for b in range(B):
    init = [np.random.normal() for i in range(6)] + [random.randint(508,600)] + [random.randint(413,500)]
    res = minimize(neg_loglikelihood_null, init, method='nelder-mead')
    if neg_loglikelihood_null(res.x) < best_lik:
        best_lik = neg_loglikelihood_null(res.x)
        best_par = res.x
    print(b)

# The MLE under the null
[b1, b2, b3, b12, b13, b23, rx, ry] = best_par

# Create B2 new populations under the null and estimate theta_hat
B2 = 1000
B3 = 2
thetahat = []

# Probabilities of ascertainment under the null (same for exposed and unexposed) 
s111 = np.exp(b1+b2+b3+b12+b13+b23)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13+b23)))
s110 = np.exp(b1+b2+b12)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13+b23)))
s101 = np.exp(b1+b3+b13)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13)))
s011 = np.exp(b2+b3+b23)/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3+b23)))
s100 = np.exp(b1)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13)))
s010 = np.exp(b2)/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3+b23)))
s001 = np.exp(b3)/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3)))
s000 = 1/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3)))

for b in range(B2):
    # Create a new population and ascertainment table
    newNx = np.random.poisson(lam = rx, size = 1)
    newNy = np.random.poisson(lam = ry, size = 1)
    newX = np.random.multinomial(newNx, [s111,s110,s101,s100,s011,s010,s001,s000])
    newY = np.random.multinomial(newNy, [s111,s110,s101,s100,s011,s010,s001,s000])
    # remove the unobserved cell
    [newx111, newx110, newx101, newx100, newx011, newx010, newx001] = newX[0:7]
    [newy111, newy110, newy101, newy100, newy011, newy010, newy001] = newY[0:7]
    obsNx = np.sum(newX[0:7])
    obsNy = np.sum(newY[0:7])
    
    # With the new observed population, estimate theta_hat and the other parameters
    def neg_loglikelihood_sim(p):
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
        
        l = (newx111*np.log(gx*p111) + newx110*np.log(gx*p110) + newx101*np.log(gx*p101) + newx011*np.log(gx*p011) + 
             newx100*np.log(gx*p100) + newx010*np.log(gx*p010) + newx001*np.log(gx*p001) + gx*(p000 - 1) +
             newy111*np.log(gy*q111) + newy110*np.log(gy*q110) + newy101*np.log(gy*q101) + newy011*np.log(gy*q011) + 
             newy100*np.log(gy*q100) + newy010*np.log(gy*q010) + newy001*np.log(gy*q001) + gy*(q000 - 1))
        
        return -l

    
    best_lik = 0
    for c in range(B3):
        init = [np.random.normal(mean, 0.1) for mean in [0, b1, b2, b3, b12, b13, b23]] + [newNx, newNy]
        res = minimize(neg_loglikelihood_sim, init, method='nelder-mead')
        if neg_loglikelihood_sim(res.x) < best_lik:
            best_lik = neg_loglikelihood_sim(res.x)
            best_par = res.x
    if best_lik < 0:
        thetahat.append(best_par[0]) 
        
    print(b)
    
# Plot the histogram of theta_hat with quantiles and observed value    
bins = np.linspace(-0.5, 0.5, 25)
plt.hist(thetahat, bins, color = 'blue')
plt.axvline(x=-1.99838417e-02, color = 'black', label = 'observed', linewidth = 3)
plt.axvline(x=np.percentile(thetahat, [2.5]), color = 'yellow', label = '2.5% quantile', linestyle = '-')
plt.axvline(x=np.percentile(thetahat, [5]), color = 'orange', label = '5% quantile', linestyle = '--')
plt.axvline(x=np.percentile(thetahat, [95]), color = 'red', label = '95% quantile', linestyle = '-.')
plt.axvline(x=np.percentile(thetahat, [97.5]), color = 'purple', label = '97.5% quantile', linestyle = ':')
plt.xlabel(r'$\hat{\theta}$')
plt.ylabel('Count')
plt.legend(loc='upper left', prop={'size': 8})
plt.show()
