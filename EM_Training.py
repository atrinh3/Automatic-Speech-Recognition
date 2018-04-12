import AlphaBeta
import Posteriors as post
import numpy as np


def probability_evidence(alpha):
    time = len(alpha)
    states = len(alpha[0])
    return alpha[time - 1][states - 1]


# Takes a set of alpha matrices for all utterances of a phrase to
# calculate the aggregate probability of evidence.  All alpha
# values are in log form so the multiplication of probabilities
# is done by simply adding the values together.
# Requires -
# alphas:   Matrix containing alpha values for all utterances.
def get_initial_likelihood(alphas):
    ell = len(alphas)
    likelihood = 0
    for epoch in range(0, ell):
        likelihood += probability_evidence(alphas[epoch])
    return likelihood


# Obtains likelihood values given a set of parameters.
# Requires-
# utterances:    The total number of utterances for iterating through MFCCs.
# mfccs:         3-Dimensional matrix containing MFCCs for all utterances.
# trans:         Guessed transition matrix to be used to calculate alpha.
# mean:          Mean values to be used to calculate alpha.
# covar:         Covariance values to be used to calculate alpha.
# iteration:     Number of iterations performed to obtain new_initial.
def get_new_likelihood(utterances, mfccs, transition, mean, covariance, iteration):
    likelihood = 0
    states = len(transition)
    for i in range(0, utterances):
        mfcc = mfccs[i]
        mfcc_len = len(mfcc)
        a = AlphaBeta.get_alpha(transition, covariance, states, mean, mfcc, mfcc_len)
        likelihood += probability_evidence(a)
    return likelihood


# This function returns a scalar describing the progress of the
# training process.  As the training functions iterate through the
# epochs, a likelihood value is computed.  After each epoch
# iteration, the likelihood list is passed into this function which
# returns the difference between the most recent likelihood value
# and the 2nd most recent.
# Requires -
# likelihood:   List of likelihood values.
def likelihood_difference(likelihood):
    length = len(likelihood)
    latest = likelihood(length - 1)
    previous = likelihood(length - 2)
    return latest - previous


# This function trains the initial state parameter.  The function will
# end if the likelihood difference between epoch iterations is small
# enough.
# Requires -
# alphas:              All alphas
# betas:               All betas
# initial_guess:       Guess to initialize the training process.
# mean_guess:          Fixed mean values
# covariance_guess:    Fixed covariance matrix
# transition_guess:    Fixed transition matrix
# mfccs:               MFCC matrix for all utterances
def train_intial_state(alphas, betas, initial_guess, mean_guess,
                       covariance_guess, transition_guess, mfccs):
    threshold = 1
    states = len(initial_guess)
    ell = len(alphas)
    new_initial = initial_guess
    likelihood = [get_initial_likelihood(alphas)]
    for epoch in range(0, ell):
        current_alpha = alphas[epoch]
        current_beta = betas[epoch]
        g = post.get_gamma(current_alpha, current_beta, 0)
        current = np.add(new_initial, g)
        tmp = get_new_likelihood(ell, mfccs, transition_guess,
                                 mean_guess, covariance_guess, epoch)
        likelihood.append(tmp)
        improvement = likelihood_difference(likelihood)
        if improvement < threshold:
            final = list(np.divide(current, epoch + 2))
            return final
    new_initial = list(np.divide(new_initial, ell + 1))
    print("Initial state training did not converge after %i iterations."
          % ell)
    print("Try a lower improvement threshold value than %.5f." % threshold)
    return new_initial


# This function builds the transition matrix.
# Requires -
# gamma:       Gamma vector
# xi:          Xi matrix
def build_transition(gamma, xi):
    states = len(gamma)
    transition = [[0] * states] * states
    for i in range(0, states):
        g = gamma[i]
        for j in range(0, states):
            x = xi[i][j]
            transition[i][j] = x / g
    return transition


# This function trains the transition matrix values.
# The process will stop once the likelihood values between iterations
# reaches a small enough value.
# Requires -
# alphas:             All alphas.
# betas:              All betas.
# mfccs:              All MFCC matrices for all utterances.
# mean_guess:         Fixed mean values.
# covariance_guess:   Fixed covariance values.
# transition_guess:   Matrix to initialize training process.
def train_transition(alphas, betas, mfccs, mean_guess, covariance_guess,
                     transition_guess):
    threshold = 1
    ell = len(alphas)
    transition = transition_guess
    states = len(alphas[0][0])
    xi = [[0] * states] * states
    gamma = [0] * states
    likelihood = [get_initial_likelihood(alphas)]
    for epoch in range(0, ell):
        mfcc = mfccs[epoch]
        mfcc_len = len(mfcc)
        alpha = alphas[epoch]
        beta = betas[epoch]
        for t in range(1, mfcc_len):
            tmp_xi = post.get_xi(alpha, beta, mfcc, t, transition, mean_guess,
                                 covariance_guess)
            tmp_gamma = post.get_gamma(alpha, beta, t)
            xi = np.add(xi, tmp_xi)
            gamma = np.add(gamma, tmp_gamma)
        new_transition = build_transition(xi, gamma)
        tmp = get_new_likelihood(ell, mfccs, new_transition, mean_guess,
                                 covariance_guess, epoch)
        likelihood.append(tmp)
        improvement = likelihood_difference(likelihood)
        if improvement < threshold:
            return new_transition
        transition = new_transition
    print("Transition matrix training did not converge after %i iterations."
          % ell)
    print("Try a lower improvement threshold value than %.5f." % threshold)
    return transition


def train_gaussian(alphas, betas, mfccs, mean_guess,
                   covariance_guess, transition):
    threshold = 1

    ell = len(alphas)
    likelihood = [get_initial_likelihood(alphas)]
    states = len(alphas[0][0])
    weighted_sum = list(np.multiply(mean_guess, 0))
    weighted_covariance = list(np.multiply(mean_guess, 0))
    gamma_sum = [0] * states
    # update mean over one epoch
    for epoch in range(0, ell):
        mfcc = mfccs[epoch]
        mfcc_t = len(mfcc)
        alpha = alphas[epoch]
        beta = betas[epoch]
        for t in range(0, mfcc_t):
            gamma = post.get_gamma(alpha, beta, t)
            for q in range(0, states):
                add = np.multiply(mfcc[t], gamma[q])
                weighted_sum = list(np.add(add, weighted_sum))
                gamma_sum[q] += gamma[q]
        new_mean = list(np.divide(weighted_sum, gamma_sum))

        for t in range(0, mfcc_t):
            gamma = post.get_gamma(alpha, beta, t)
            for q in range(0, states):
                variance = np.subtract(mfcc[t], new_mean[q])
                variance = list(np.multiply(variance, variance))
                add = np.multiply(variance, gamma[q])
                weighted_covariance = list(np.add(add, weighted_covariance))
        new_covariance = list(np.divide(weighted_covariance, gamma_sum))
        new_like = get_new_likelihood(ell, mfccs, transition, new_mean,
                                      new_covariance, epoch)
        likelihood.append(new_like)
        improvement = likelihood_difference(likelihood)
        if improvement < threshold:
            return new_mean, new_covariance
    new_mean = list(np.divide(weighted_sum, gamma_sum))
    new_covariance = list(np.divide(weighted_covariance, gamma_sum))
    print("Gaussian parameter training did not converge after %i iterations."
          % ell)
    print("Try a lower improvement threshold value than %.5f." % threshold)
    return new_mean, new_covariance


# # This function trains the mean parameter.
# # Requires -
# # alphas:             All alphas.
# # betas:              All betas.
# # mfccs:              All MFCC matrices for all utterances.
# # mean_guess:         Initialize mean values for training.
# # covariance_guess:   Fixed covariance values
# # transition_guess:   Fixed transition matrix
# def train_mean(alphas, betas, mfccs, mean_guess, covariance_guess,
#                transition_guess):
#     threshold = 1
#     ell = len(alphas)
#     states = len(alphas[0][0])
#     likelihood = [get_initial_likelihood(alphas)]
#     weighted_sum = list(np.multiply(mean_guess, 0))
#     gamma_sum = [0] * states
#     for epoch in range(0, ell):
#         mfcc = mfccs[epoch]
#         alpha = alphas[epoch]
#         beta = betas[epoch]
#         mfcc_t = len(mfcc)
#         for q in range(0, states):
#             for t in range(0, mfcc_t):
#                 gamma = post.get_gamma(alpha, beta, t)
#                 gamma_q = gamma[q]
#                 tmp = np.multiply(mfcc[t], gamma_q)
#                 weighted_sum[q] = list(np.add(tmp, weighted_sum[q]))
#                 gamma_sum[q] += gamma_q
#         new_mean = list(np.divide(weighted_sum, gamma_sum))
#         tmp = get_new_likelihood(ell, mfccs, transition_guess, new_mean,
#                                  covariance_guess, epoch)
#         likelihood.append(tmp)
#         improvement = likelihood_difference(likelihood)
#         if improvement < threshold:
#             return new_mean
#     new_mean = list(np.divide(weighted_sum, gamma_sum))
#     return new_mean
#
#
# # This function trains the covariance parameter. This happens
# # separately from the function that trains the mean.
# # Requires -
# # alphas:             All alphas.
# # betas:              All betas.
# # mfccs:              All MFCC matrices for all utterances.
# # mean_guess:         Fixed mean values.
# # covariance_guess:   Initial covariance values for training.
# # transition_guess:   Fixed transition matrix
# def train_covariance(alphas, betas, mfccs, means, covariance_guess,
#                      transition_guess):
#     threshold = 1
#     ell = len(alphas)
#     states = len(alphas[0][0])
#     likelihood = [get_initial_likelihood(alphas)]
#     weighted_sum = list(np.multiply(covariance_guess, 0))
#     gamma_sum = 0
#     for epoch in range(0, ell):
#         alpha = alphas[epoch]
#         beta = betas[epoch]
#         mfcc = mfccs[epoch]
#         mfcc_t = len(mfcc)
#         for q in range(0, states):
#             mean_q = means[q]
#             for t in range(0, mfcc_t):
#                 var = np.subtract(mfcc[t], mean_q)
#                 var = list(np.multiply(var, var))
#                 gamma = post.get_gamma(alpha, beta, t)
#                 gamma_q = gamma[q]
#                 gamma_sum += gamma_q
#                 tmp = list(np.multiply(var, gamma_q))
#                 weighted_sum[q] = list(np.add(weighted_sum[q], tmp))
#         new_covariance = list(np.divide(weighted_sum, gamma_sum))
#         tmp = get_new_likelihood(ell, mfccs, transition_guess, means,
#                                  new_covariance, epoch)
#         likelihood.append(tmp)
#         improvement = likelihood_difference(likelihood)
#         if improvement < threshold:
#             return new_covariance
#     new_covariance = list(np.divide(weighted_sum, gamma_sum))
#     return new_covariance
