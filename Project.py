import numpy as np


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
	
	
def prob_observation(mean, covariance, mfcc):
    # mean and covariance describes the state
    n = len(mean)
    product = 1
    for i in range(0, n):
        product = product * covariance[i]
    denominator = ((2*np.pi)**(n/2) * np.sqrt(product))
    left = 1/denominator
	var = mfcc - mean
    sum = 0
    for i in range(0, n):
        sum += var[i]**2 / covariance[i]
    right = np.exp(-.5 * sum)
    return left * right


def get_mean(hmm):
    mean = []
    t = len(hmm)
    q = len(hmm[0])
    for i in range(0, q):
        sum = 0
        for j in range(0, t):
            sum += hmm[i][j]
        mean.append(sum)
    return mean


def get_covariance(hmm, mean):
    # hmm is a matrix of dimension N x Tl where Tl = length of all
    # utterances combined
    t = len(hmm)
    q = len(hmm[0])
    covariance = []
    for i in range(0, q):
        sum = 0
        for j in range(0, t):
            sum += (hmm[i][j] - mean[i]) ** 2
        covariance.append(sum / q)
    return covariance


def prob_evidence(alpha, beta):
    # p(x[1:t])
    evidence = []
    for t in range(0, len(alpha)):
        evidence.append(np.dot(alpha[t], beta[t]))
    return evidence


# Build this get_gamma function into the get_alpha function
def get_gamma(alpha, beta, state):
    out = np.dot(alpha, beta)
    return alpha[state] * beta[state] / out


def get_xi(alpha, beta, transition, covariance, mfcc, t, state):
    # p(observation) * beta[q][t] * p(transition) * alpha[q-1][t-1]
	state = STATE
	po = prob_observation(mean, covariance, mfcc)
	trans = transition[state(t)][state(t-1)]
	b = beta[state][t]
	a = alpha[state-1][t-1]
	return po * b * trans * a


# START
l = 20
hmm = []
mean = get_mean(hmm)
initial_covariance = get_covariance(hmm, mean)
initial_mean = mean + np.random.normal(0, 
                                       np.mean(initial_covariance)/8, 
                                       len(mean))

# L T Q
# log likelihood is alpha matrix summed at the last t = T
# p(x[1:T]|w) is the last column of the alpha matrix
# each state gaussian should be represented by a mean and a covariance
