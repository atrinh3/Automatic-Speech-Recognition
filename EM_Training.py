def check_convergence(before, after):
    h = len(before)
    w = len(before[0])
    sum = 0
    compare = 0
    for i in range(0, h):
        for j in range(0, w):
            diff = after[i][j] - before[i][j]
            compare += before[i][j]
            sum += diff
    return sum / compare

    
def train_intial_state(alphas, betas, initial_guess, mean_guess, 
                       covariance_guess, transition_guess):
    ell = len(alphas)
    final_intial = initial_guess
    for i in range(0, ell):
        current_alpha = alphas[i]
        current_beta = betas[i]
        g = get_gamma(current_alpha, current_beta, 0)
        
        tmp = list(np.divide(final_initial, i + 1))
        final_initial = list(np.add(final_initial, g))
        final_initial = list(np.divide(sum, i + 2)
        check = check_convergence(tmp, final_initial)
        if check < threshold:
            return final_initial
    return final_initial


def train_transition():


def train_mean():


def train_covariance():






















