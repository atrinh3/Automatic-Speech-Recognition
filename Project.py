import numpy as np
import matplotlib.pyplot as plt



def get_transition(states):
    mat = [[0] * states] * states
    for i in range(0, states - 1):
        mat[i][i] = .5
        mat[i][i + 1] = .5
        mat[states][states] = 1
    # debugging
    plt.imshow(mat)
    plt.show()
    return mat


def prob_evidence(alpha, beta):
    # p(x[1:t])
    evidence = []
    for t in range(0, len(alpha)):
        evidence.append(np.dot(alpha[t], beta[t]))
    return evidence




# L T Q
# log likelihood is alpha matrix summed at the last t = T
# p(x[1:T]|w) is the last column of the alpha matrix
# each state gaussian should be represented by a mean and a covariance


def generate_gaussian(covariance_array, mean):
    gauss = np.ndarray
    covariance_matrix = np.identity(len(covariance_array)) * covariance_array
    return gauss


# ========================================================================
# ========================================================================

# Equation from lecture 9, F58/77 & hw5.pdf equation (21)
def prob_observation(mean, covariance, mfcc, states):
    # mean and covariance describes the state
    observation_matrix = []
    product = 1
    for i in range(0, states):
        product = product * covariance[i]
    left = 1 / (((2 * np.pi) ** (states / 2)) * np.sqrt(product))
    for t in range(0, len(mfcc)):
        mfcc_vector = mfcc[t]
        observation_vector = []
        for q in range(0, states):
            noise = np.random.normal(0, 0.01, len(mean))
            tmp = noise + mean
            var = mfcc_vector - tmp
            sum = 0
            for i in range(0, states):
                sum += var[i] ** 2 / covariance[i]
            right = np.exp(-.5 * sum)
            observation_vector.append(left * right)
        observation_matrix.append(observation_vector)
    return observation_matrix
    # probabilty of having MFCC fector at time t for each state q


def get_mean(mfcc):
    mean = []
    t = len(mfcc)
    n = len(mfcc[0])
    for i in range(0, n):
        sum = 0
        for j in range(0, t):
            sum += mfcc[i][j]
        mean.append(sum / t)
    return mean


def get_covariance(mfcc, mean):
    # hmm is a matrix of dimension N x Tl where Tl = length of all
    # utterances combined
    t = len(mfcc)
    n = len(mfcc[0])
    covariance = []
    for i in range(0, n):
        sum = 0
        for j in range(0, t):
            sum += (mfcc[i][j] - mean[i]) ** 2
        covariance.append(sum / t)
    return covariance


# Equation from lecture 9, F9/77
def get_alphas(passed_observation, passed_trans, passed_mfcc, states, t):
    if t < 2:
        # Reached beginning of the HMM.  Need to end the recursion
        # At the first time step, the algorithm will initialize to
        # the first state meaning q1 = 1 and all other q's = 0.
        alpha_initial = [0] * len(passed_trans[0])
        alpha_initial[0] = 1
        return [alpha_initial]
    last_alpha = get_alphas(passed_observation,
                            passed_trans,
                            passed_mfcc,
                            states,
                            t - 1)
    current_alpha = []
    for i in range(0, states):
        sum = 0
        for j in range(0, states):
            # i = TO state, j = FROM state
            transition_probability = passed_trans[j][i]
            sum += transition_probability * last_alpha[j][t]
        observation = passed_observation[i][t]
        current_alpha.append(observation * sum)
    return current_alpha.extend(last_alpha)
    # S/B able to get gammas simultaneously from alpha recursions


# Equation from lecture 9, F14/77
def get_betas(passed_observation, passed_covar,
              passed_trans, passed_mfcc, states, t):
    if t > len(passed_mfcc) - 1:
        last_beta = [1] * len(passed_trans)
        return [last_beta]
    next_beta = get_betas(passed_observation, passed_covar,
                          passed_trans, passed_mfcc, states, t + 1)
    current_beta = []
    for i in range(0, states):
        sum = 0
        for j in range(0, states):
            # i = FROM state, j = TO state
            transition_probability = passed_trans[i][j]
            sum += passed_covar[j] * transition_probability * next_beta[j][0]
        current_beta.append(sum)
    return current_beta.extend(next_beta)


# Equation from lecture 9, F19/77
def get_gamma(alpha, beta, state):
    gamma_matrix = []
    for t in range(0, len(alpha)):
        tmp = np.dot(alpha[t], beta[t])
        gamma_vector = []
        for q in range(0, state):
            gamma_vector.append(alpha[q][t] * beta[q][t] / tmp)
        gamma_matrix.append(gamma_vector)
    return gamma_matrix


# Equation in lecture 9, F21/77
def get_xi(alpha, beta, observation, transition, covariance, mfcc, states):
    # p(observation) * beta[q][t] * p(transition) * alpha[q-1][t-1]
    xi_matrix = []
    for t in range(0, len(alpha)):
        xi_vector = []
        for q in range(0, states):
            po = observation[q][t]
            b = beta[q][t]
            trans_a = transition[q][q - 1]
            trans_b = transition[q][q]
            a = alpha[q][t - 1]
            xi_vector.append(po * b * trans_a * a)
        xi_matrix.append(xi_vector)
    return xi_matrix


# =============#
#     START    #
# -------------#
mfcc = []
mfcc_mean = get_mean(mfcc)
covariance = get_covariance(mfcc, mfcc_mean)

odessa_states = 10
odessa_mean = [1 / odessa_states] * odessa_states
odessa_trans = np.matrix([[0.5, 0.5, 0, 0, 0, 0],
                          [0, 0.5, 0.5, 0, 0, 0],
                          [0, 0, 0.5, 0.5, 0, 0],
                          [0, 0, 0, 0.5, 0.5, 0],
                          [0, 0, 0, 0, 0.5, 0.5],
                          [0, 0, 0, 0, 0, 1]])
mfcc_observation = prob_observation(odessa_mean, covariance, mfcc, odessa_states)

alpha_t = len(mfcc)
odessa_alphas = get_alphas(mfcc_observation, odessa_trans,
                           mfcc, odessa_states, alpha_t)
odessa_betas = get_betas(mfcc_observation, covariance, odessa_trans, mfcc, odessa_states, 0)



# Write code for the data structures & algorithms of the HMMs


# Write code that computes the alpha recursion.
# Probably want to store, for each HMM, the entire alpha matrix for each word w.


# Write code that computes the beta recursion.
# Probably want to store for each HMM, the entire beta matrix for each word w.


# Create a separate alpha and beta matrix for EACH utterance of each phrase.
# Use the alpha and beta matrix to compute gamma and epsilon quantities


# From either alpha or beta matrix,
# compute probability of (x1, x2, ...,  xT) given w (p(x1:T)|w)


# Write code that computes gamma recursion.  Doesn't need to be stored since
# it's easily computed from alpha and beta matrices.


# Write code that computes epsilon quantity.  Doesn't need to be stored
# since it's easily computed from alpha and beta matrices.


# Collect set of training utterances for each word.


# Implement the EM algorithm for HMMs.


# Use the trained HMMs computed in previous steps


# ASR





