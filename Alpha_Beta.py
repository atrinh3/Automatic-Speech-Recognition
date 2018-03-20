# Calculates the sum of 2 probabilities when the available numbers
# are the probability values in log form.  Since there is a
# representation of -inf, if the difference between the numbers
# being added is greater than the representation of inf, this 
# function will simply return the larger value without performing 
# the addition.
# Requires -
# a:    The log form of probability a
# b:    The log form of probability b
# low:  The scalar representation of -inf when a probability = 0
def log_add(a, b, low)
    small = min(a, b)
    big = max(a, b)
    diff = a - b
    if diff < -(math.log(-low)):
        # big >> small
	    return big
    c = b + math.log(1 + math.exp(a - b)
    return c


# Customized log call to allow for log(probabilities)
# Since probabilities will always be >= 0, the only error would 
# come when the probability = 0 which would cause the log call to 
# return -inf.  In this function, attempting to call log(0) would
# return a pre-specified number that will represent -inf.
# Requires - 
# number:   The number the function is taking the log of.
# log_low:  The scalar representation of -inf when 'number' = 0
def get_log(number, log_low):
    if number == 0:
        return log_low
    return math.log(number)
	

# Calculate alpha matrix using full MFCC matrix & log(probabilities)
# Requires - 
# transition:   QxQ transition matrix
# covariances:  QxN covariance matrix
# states:       Q - number of states
# means:        QxN mean matrix
# mfcc:         TxN sideways MFCC matrix
# t:            current time location in 0<t<T
def get_alpha(transitions, covariances, states, means, mfcc, t):
    if t < 1:
        # When t = 0, the chance of being in the first state is
        # 100% so the whole column is 0 except for the first state
        # which will be set to 1.  For log probabilities, 0's are
        # represented with -10e30 since log(0) = -inf.  Likewise,
        # the 1 will be represented with a 0 since log(1) = 0.
        low_log = -10e30
        initial_alpha = [low_log] * states
        initial_alpha[0] = math.log(1)
        return [initial_alpha]
    current_mfcc = mfcc[t]
    alpha_matrix = get_alpha(transitions, 
                             covariances, 
                             states, 
                             means,
                             mfcc, 
                             t-1)
    previous_alpha = alpha_matrix[t - 1]
    build_alpha = [0] * states
    low_log = -10e30
    for q in range(0, states):
        local_distortion = observation_probability(current_mfcc, 
                                                   means, 
                                                   covariances)
        local_distortion = get_log(local_distortion)
        sum = log_low
        for r in range(0, states):
		    state_transition = transition[r][q]
		    state_transition = get_log(state_transition)
		    alpha_r = previous_alpha[r]
		    product = alpha_r + state_transition
		    sum = log_add(sum, product, log_low)
        build_alpha[q] = local_distortion + sum
    return alpha_matrix
	
	
# Calculate beta matrix using full MFCC matrix & log(probabilities)
# Requires - 
# transition:   QxQ transition matrix
# covariances:  QxN covariance matrix
# states:       Q - number of states
# means:        QxN mean matrix
# mfcc:         TxN sideways MFCC matrix
# t:            current time location in 0<t<T
def get_beta(transition, covariances, states, means, mfcc, t):
    if t > len(mfcc) - 1:
	    # When t = T-1, the chance of being in the last state is
	    # 100% so the whole column is 0 except for the last state
	    # which will be set to 1.  For log probabilities, 0's are
	    # represented with -10e30 since log(0) = -inf.  Likewise,
	    # the 1 will be represented with a 0 since log(1) = 0.    
	    log_low = -10e30
	    last_beta = [log_low] * states
	    last_beta[states - 1] = math.log(1)
	    return [last_beta]
    beta_matrix = get_beta(transition, 
                           covariances, 
                           states, 
                           means, 
                           mfcc, 
                           t + 1)
    next_mfcc = mfcc[t + 1]
    build_beta = [0] * states
    next_beta = beta_matrix[0]
    log_low = -10e30
    for q in range(0, states):
        sum = log_low
        for r in range(0, states):
            state_transition = transition[q][r]
            state_transition = get_log(state_transition)
            next_distortion = observation_probability(next_mfcc, 
                                                      means, 
                                                      covariances)
            next_distortion = get_log(next_distortion)
            beta_r = next_beta[r]
            product = beta_r + next_distortion + state_transition
            sum = log_add(sum, product, log_low)
        build_beta[q] = sum
    return build_beta
