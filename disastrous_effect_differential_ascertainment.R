# Number of bootstrap populations
B = 100

# Mean size of exposed and unexposed populations
gammaE = 500
gammaU = 1000

# Main effects a_i. The probability for each individual of not being 
# ascertained only depends on the main effects
a1 = 0.5
a2 = 0.5
a3 = 0.5
a4 = 0.5
a5 = 0.5

# Possible values of differential ascertainment
thetas = c(-1, -0.5, 0, 0.5, 1)

# Create empty matrices for means, upper and lower bounds on odds ratio
mean1 = matrix(NA, ncol = 5, nrow = 1)
up1 = matrix(NA, ncol = 5, nrow = 1)
low1 = matrix(NA, ncol = 5, nrow = 1)
mean3 = matrix(NA, ncol = 5, nrow = 1)
up3 = matrix(NA, ncol = 5, nrow = 1)
low3 = matrix(NA, ncol = 5, nrow = 1)
mean5 = matrix(NA, ncol = 5, nrow = 1)
up5 = matrix(NA, ncol = 5, nrow = 1)
low5 = matrix(NA, ncol = 5, nrow = 1)

for(i in 1:5){
  theta = thetas[i]
  # probabilities of not observing an individual with the set
  # level of differential ascertainment
  pE1_miss = 1/(1+exp(theta+a1))
  pU1_miss = 1/(1+exp(a1))
  pE3_miss = 1/((1+exp(theta+a1))*(1+exp(theta+a2))*(1+exp(theta+a3)))
  pU3_miss = 1/((1+exp(a1))*(1+exp(a2))*(1+exp(a3)))
  pE5_miss = 1/((1+exp(theta+a1))*(1+exp(theta+a2))*(1+exp(theta+a3))**(1+exp(theta+a5))**(1+exp(theta+a5)))
  pU5_miss = 1/((1+exp(a1))*(1+exp(a2))*(1+exp(a3))**(1+exp(a4))**(1+exp(a5)))
  ratio1 = c()
  ratio3 = c()
  ratio5 = c()
  for(b in 1:B){
    # Poisson population with mean gamma
    NE = rpois(1, gammaE)
    NU = rpois(1, gammaU)
    # Counts in the missing cells
    NE1_miss = rbinom(1, NE, pE1_miss)
    NU1_miss = rbinom(1, NU, pU1_miss)
    NE3_miss = rbinom(1, NE, pE3_miss)
    NU3_miss = rbinom(1, NU, pU3_miss)
    NE5_miss = rbinom(1, NE, pE5_miss)
    NU5_miss = rbinom(1, NU, pU5_miss)
    # Ratios of observed exposed individuals and observed unexposed individuals
    ratio1 = c(ratio1, (NE-NE1_miss)/(NU-NU1_miss))
    ratio3 = c(ratio3, (NE-NE3_miss)/(NU-NU3_miss))
    ratio5 = c(ratio5, (NE-NE5_miss)/(NU-NU5_miss))
  }
  mean1[1,i] = mean(ratio1)
  up1[1,i] = quantile(ratio1, 0.975)
  low1[1,i] = quantile(ratio1, 0.025)
  mean3[1,i] = mean(ratio3)
  up3[1,i] = quantile(ratio3, 0.975)
  low3[1,i] = quantile(ratio3, 0.025)
  mean5[1,i] = mean(ratio5)
  up5[1,i] = quantile(ratio5, 0.975)
  low5[1,i] = quantile(ratio5, 0.025)
  print(i)
}

low1
mean1
up1

low3
mean3
up3

low5
mean5
up5

