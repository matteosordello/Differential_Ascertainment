import numpy as np
import pandas as pd
import scipy.stats
import csv
from scipy.stats import multinomial
from scipy.stats import poisson
from scipy.optimize import minimize
import math
import random

# number of bootstrap populations
B = 200
# number of initialization for estimating the parameters
B2 = 3
# matrix to store the best parameters for each iteration
p = np.zeros((B, 9))

# choose the value for differential ascertainment (theta)
the = 0
# main effects
b1 = 0.5
b2 = 0.5
b3 = 0.5
# interaction effects
b12 = -0.2
b13 = -0.2
b23 = -0.2
# means of exposed and unexposed populations
gE = 500
gU = 1000
# ascertainment probabilities (r for the exposed, s for the unexposed)
r111 = np.exp(b1+b2+b3+b12+b13+b23+3*the)/((1+np.exp(b1+the))*(1+np.exp(b2+b12+the))*(1+np.exp(b3+b13+b23+the)))
r110 = np.exp(b1+b2+b12+2*the)/((1+np.exp(b1+the))*(1+np.exp(b2+b12+the))*(1+np.exp(b3+b13+b23+the)))
r101 = np.exp(b1+b3+b13+2*the)/((1+np.exp(b1+the))*(1+np.exp(b2+b12+the))*(1+np.exp(b3+b13+the)))
r011 = np.exp(b2+b3+b23+2*the)/((1+np.exp(b1+the))*(1+np.exp(b2+the))*(1+np.exp(b3+b23+the)))
r100 = np.exp(b1+the)/((1+np.exp(b1+the))*(1+np.exp(b2+b12+the))*(1+np.exp(b3+b13+the)))
r010 = np.exp(b2+the)/((1+np.exp(b1+the))*(1+np.exp(b2+the))*(1+np.exp(b3+b23+the)))
r001 = np.exp(b3+the)/((1+np.exp(b1+the))*(1+np.exp(b2+the))*(1+np.exp(b3+the)))
r000 = 1/((1+np.exp(b1+the))*(1+np.exp(b2+the))*(1+np.exp(b3+the)))
s111 = np.exp(b1+b2+b3+b12+b13+b23)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13+b23)))
s110 = np.exp(b1+b2+b12)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13+b23)))
s101 = np.exp(b1+b3+b13)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13)))
s011 = np.exp(b2+b3+b23)/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3+b23)))
s100 = np.exp(b1)/((1+np.exp(b1))*(1+np.exp(b2+b12))*(1+np.exp(b3+b13)))
s010 = np.exp(b2)/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3+b23)))
s001 = np.exp(b3)/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3)))
s000 = 1/((1+np.exp(b1))*(1+np.exp(b2))*(1+np.exp(b3)))
for b in range(0, B):
    # generate size of the population
    nE = np.random.poisson(gE)
    nU = np.random.poisson(gU)
    # generate ascertainment matrices with multinomial distribution
    myXw = np.random.multinomial(nE, [r111,r110,r101,r011,r100,r010,r001,r000])
    myXb = np.random.multinomial(nU, [s111,s110,s101,s011,s100,s010,s001,s000])
    # remove unobserved counts
    [newx111, newx110, newx101, newx011, newx100, newx010, newx001] = myXw[0:7]
    [newy111, newy110, newy101, newy011, newy100, newy010, newy001] = myXb[0:7]
    # define counts of observed individuals
    newNx = np.sum(myXw[0:7])
    newNy = np.sum(myXb[0:7])

    # Now from the observed matrices we try to estimate the parameters and the 
    # underlying true counts of the populations, using the negative log-likelihood.
    # The parameters are:
    # t -> differential ascertainment (theta)
    # a_i -> main effects
    # a_ij -> interactions
    # g_x -> count of exposed individuals
    # g_y -> count of unexposed individuals
    def my_neg_likelihood(p):
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

        # the log likelihood is:
        l = (newx111*np.log(gx*p111) + newx110*np.log(gx*p110) + newx101*np.log(gx*p101) + newx011*np.log(gx*p011) + 
             newx100*np.log(gx*p100) + newx010*np.log(gx*p010) + newx001*np.log(gx*p001) + gx*(p000 - 1) +
             newy111*np.log(gy*q111) + newy110*np.log(gy*q110) + newy101*np.log(gy*q101) + newy011*np.log(gy*q011) + 
             newy100*np.log(gy*q100) + newy010*np.log(gy*q010) + newy001*np.log(gy*q001) + gy*(q000 - 1))
        
        return -l
    
    # Here we use a built-in optimizer to find the MLE of the parameters
    best_lik = 0
    # We start the optimization in B2 different points centered around the true value
    for c in range(0, B2):
        init = np.random.normal([the, 0.5, 0.5, 0.5, -0.2, -0.2, -0.2],0.1).tolist() + np.random.uniform([490, 990],[510, 1010]).tolist()
        res = minimize(my_neg_likelihood, init, method='nelder-mead')
        if my_neg_likelihood(res.x) < best_lik:
            best_lik = my_neg_likelihood(res.x)
            best_par = res.x
            
    print(b)
    # Store the best estimates
    p[b,:] = best_par
    
# print the means and confidence intervals for each parameter
np.mean(p, axis=0)
np.percentile(p, [2.5, 97.5], axis=0)
