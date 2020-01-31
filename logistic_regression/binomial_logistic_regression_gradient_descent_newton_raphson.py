## Binomial Logistic Regression using Newton Raphson

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#### Function Definitions
def compute_log_likelihood(y, est_prob): # est_prob: Estimated probability # Includes conversion from probability to log(odds)
    logit = ((y*est_prob) + ((1-y)*(1-est_prob)))
    return np.prod(np.where(logit != 0, logit, 1))

def net_input(beta, x):
    # compute the weighted sum of inputs
    return x.dot(beta)

def log_odds_to_prob(x): 
    # Conversion from log(odds) to probability
    return np.divide(np.exp(x), (1 + np.exp(x)))

def compute_probabilities(beta, x): 
    # Calculate the value of est_probs (predictions on each observation) given x(input) and estimated betas
    return log_odds_to_prob(net_input(beta, x))

def logistic(x_unbiased, y, learningrate, dif):
    # Bias variable 
    bias_variable = np.ones(len(x_unbiased))
    # Plug input along with bias_variable
    x = np.column_stack([bias_variable, x_unbiased])
    # Initialize parameterss/coefficients
    beta = np.array([0] * x.shape[1])
    # Container to store log likelihoods and derivatives
    log_likelihoods = []
    derivatives = []
    diff = 10000 # Any high value
    while (diff > dif):
        est_probs = compute_probabilities(beta, x) # Predict with new value of beta
        W_matrix = np.diag((est_probs) * (1-est_probs)) # A sparse diagonal matrix
        # Assume x_new to have dimensions (n x f), whereas y, pi to have dimensions (n x 1)
        # solve(t(x_new)%*%W : factor in pi with x_new (x_new.T * (identity_matrix * (pi[i]*(1-pi[i]))) : outputs (f x n)
        # (solve(t(x_new)%*%W%*%as.matrix(x_new))): factor in each feature of our computed matrix(x_new factored with estimated probability) with each feature of orginal input matrix : outputs (f x f)
        # (t(x_new)%*%(y - pi)) : Factor in error with each of the feature column with respect to each example : outputs (f x 1)
        derivative = np.dot(np.linalg.inv(np.dot(np.dot(x.T, W_matrix), x)), np.dot(x.T, (y - est_probs))) # This computation is equivalent to computing derivative w.r.t to each feature, more about (residuals x feature level transformation)
        beta = beta + derivative # We add, because we are doing gradient ascent(to maximize the likelihood)
        diff = np.sum(derivative.flatten()**2)
        log_likelihood = compute_log_likelihood(y, est_probs)
        # Append Log likelihood and derivative into the respective container
        log_likelihoods.append(log_likelihood)
        derivatives.append(derivative)
    return beta, log_likelihoods, derivatives

x = np.arange(1, 11)
y = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])

print(x.shape, y.shape)

beta, log_likelihoods, derivatives = logistic(x, y, 0.01, 0.000000001)

print(f'Log-Likelihood: {log_likelihoods[-1]}')