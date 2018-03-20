# This function calculates a Qx1 vector given alpha, beta and time t.
# The input alpha and beta matrices are already in log form.
# Prior to making the gamma calculations, the log form of both the
# alpha and beta values have to be undone.  At this point, it
# should be safe to unlog the alpha and beta values without
# experiencing underflow errors.
# Required -
# alpha:  QxT matrix of log'd alpha values
# beta:   QxT matrix of log'd beta values
# t:      Time value where 0 < t < T
def get_gamma(alpha, beta, t):
    states = len(alpha[0])
	alpha_t = alpha[t]
	beta_t = beta[t]
    sum = 0
    for q in range(0, states):
        a_q = math.exp(alpha_t[q])
        b_q = math.exp(beta_t[q])
        sum += a_q * b_q
    gamma = [0] * states
    for q in range(0, states):
        a_q = math.exp(alpha[q])
        b_q = math.exp(beta[q])
        gamma[q] = a_q * b_q / sum
    return gamma
	
	
# This function calculates the xi matrix from alpha, beta, & time t.
# The input parameters transition, means, and covariances are given
# to allow calculation of the observation probability.  When using
# the alpha and beta values, it is necessary to unlog the values.
# Requires -
# alpha:        QxT alpha matrix
# beta:         QxT beta matrix
# mfcc:         Full TxN MFCC matrix
# t:            Time step t where 0 < t < T
# transition:   QxQ transition matrix
# means:        QxN mean matrix for calculating p(observation)
# covariances:  QxN covariance matrix for calculating p(observation)
def get_xi(alpha, beta, mfcc, t, transition, means, covariances):
    states = len(alpha[0])
	alpha_t = alpha[t - 1]
	beta_t = beta[t]
    mfcc_t = mfcc[t]
	xi_matrix = [0] * states
	for q in range(0, states):
	    beta_q = beta_t[q]
		beta_q = math.exp(beta_q)
		local_distortion = observation_probability(mfcc_t, means, covariances)
	    r_vect = [0] * states
		for r in range(0, states):
            state_transition = transition[r][q]
            alpha_r = alpha_t[r]
            alpha_r = math.exp(alpha_r)
            r_vect[r] = local_distortion * beta_q * state_transition * alpha_r
        xi_matrix[q] = r_vect
    return xi_matrix
