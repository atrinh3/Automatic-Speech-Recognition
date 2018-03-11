# import matplotlib.pyplot as plt
import scipy.io.wavfile as wv
import sounddevice as sd
import numpy as np
import math


# L T Q
# log likelihood is alpha matrix summed at the last t = T
# p(x[1:T]|w) is the last column of the alpha matrix
# each state gaussian should be represented by a mean and a covariance

# ========================================================================
# ========================================================================


# The "make_window" function takes a window width parameter and returns
# an array representing the Hamming window
def make_window(w, freq):
    sample_size = int(freq * w / 1000)
    ham = []
    for n in range(0, sample_size):
        ham.append(0.54 - 0.46 * math.cos(2 * math.pi * n / sample_size))
    return ham


def make_mel_banks(min, bins, fs):
    mel_min = 1127 * math.log1p(1 + min/700)
    mel_max = 1127 * math.log1p(1 + fs/700)
    mel_banks = [float(min)]
    for i in range(1, bins+2):
        mel_banks.append(i * (mel_max - mel_min) / bins)
    # inverse mel
    count = 0
    for i in mel_banks:
        mel_banks[count] = 700 * (np.exp(i / 1125) - 1)
        count += 1
    return mel_banks


def make_triangle(min, max, size, nfft, maxf):
    low = int(math.floor((nfft + 1) * min / maxf))
    high = int(math.floor((nfft + 1) * max / maxf))
    mid = int((low + high) / 2)
    triangle = []
    for i in range(0, low):
        triangle.append(0)
    for i in range(low, mid):
        id = len(triangle)
        triangle.append((id - low) / (mid - low))
    for i in range(mid, high):
        id = len(triangle)
        triangle.append((high - id) / (high - mid))
    for i in range(high, size):
        triangle.append(0)
    return triangle


def mat_multiply(a, b):
    sum = 0
    for i in range(0, len(a)):
        sum += a[i] * b[i]
    return sum


def mel_filter(data, banks, nfft):
    filtered = []
    maxf = max(banks)
    for i in range(2, len(banks)):
        lo = banks[i-2]
        hi = banks[i]
        tri = make_triangle(lo, hi, len(data), nfft, maxf)
        coefficient = mat_multiply(data, tri)
        filtered.append(coefficient)
    # 4) take log of result
    for i in range(0, len(filtered)):
        filtered[i] = math.log10(abs(filtered[i]))
    return filtered


# have a windowed array of data points now
def get_vector(array, fs, max):
    # 1) compute FFT- Make FFT length= to next power of 2 above window length
    my_pad = int(np.power(2, 0 + np.ceil(np.log2(len(array)))))
    out = np.fft.fft(array, my_pad)

    # 2) take magnitude of FFT (throw away phase)
    out = abs(out)

    # create mel banks
    min = 0
    bins = 23
    mels = make_mel_banks(min, bins, fs)

    # 3) v1 compute mel filter warped spectra
    # take half of array
    tmp = len(out) / 2 + 1
    tmp = out[0:int(tmp)]

    # 5) take IDFT / DCT
    out = np.fft.irfft(mel_filter(tmp, mels, my_pad/2))

    # 6) retain first 13 coefficients
    return out[0:max+1]


def produce_mfcc(signal, window, fs, fr, max):
    # have a window function
    # want to apply window to signal
    # perform cepstrum operations to windowed signal
    # repeat with window shifted
    window_length = len(window)
    signal_length = len(signal)
    shift = 1 / fr * fs
    mfcc = []
    i = 0
    count = 0
    while i + window_length <= signal_length:
        # Get truncated sample and multiply with window function
        start = int(i)
        end = int(i + window_length)
        tmp = signal[start:end]
        tmp = np.multiply(tmp, window)
        mfcc.append(get_vector(tmp, fs, max))
        i += shift
        count += 1
    return mfcc


def get_delta(data, diameter):
    length = len(data)
    d = []
    for i in range(diameter, length - diameter):
        num = 0
        den = 0
        for j in range(-diameter, diameter + 1):
            a1 = data[i + j]
            if diameter <= j < length - diameter:
                num += np.multiply(j, a1)
                den += j * j
                d.append(num/den)
    return d


# Initializing transition probabilities so that every transition that's not
# maintaining state or going to the next state is = 0.
def get_transition(states):
    mat = np.ndarray([states, states]) * 0
    for i in range(0, states - 1):
        mat[i][i] = np.random.normal(0.5, .05, 1)
        mat[i][i + 1] = 1 - mat[i][i]
        # mat[i][i] = .5
        # mat[i][i + 1] = .5
    mat[states - 1][states - 1] = 1
    # debugging
    # plt.imshow(mat)
    # plt.show()
    return mat


# Equation from lecture 9, F58/77 & hw5.pdf equation (21)
# mean:        - Mean vector including the global mean values
# covariance:  - 1xN dim. matrix containing global covariance values
# mfcc:        - MFCC matrix of utterance
# states:      - Number of states in the HMM
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
            noise = np.random.normal(0, 0.005, len(mean))
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


def prob_evidence(alpha, beta):
    # p(x[1:t])
    result =  []
    for t in range(0, len(alpha)):
        result.append(np.dot(alpha[t], beta[t]))
    return result


# Equation from lecture 9, F58/77 & hw5.pdf equation (21)
# mean:        Mean vector including the global mean values
# covariance:  1xN dim. matrix containing global covariance values
# mfcc:        Single MFCC vector
def prob_observation_v2(mean, covariance, mfcc):
    # mean and covariance describes the state
    length = len(mean)
    product = 1
    for i in range(0, length):
        product = product * covariance[i]
    left = 1 / (((2 * np.pi) ** (length / 2)) * np.sqrt(product))
    diff = np.subtract(mfcc, mean)
    sum = 0
    for i in range(0, length):
        sum += (diff[i] ** 2) / covariance[i]
    right = np.exp(-0.5 * sum)
    return left * right


# Returns a vector consisting of the mean of each element of the MFCC
def get_mean(mfcc):
    mean = []
    t = len(mfcc)
    n = len(mfcc[0])
    for i in range(0, n):
        m = np.mean(mfcc[i])
        mean.append(m)
    return mean


def get_covariance(mfcc, mean):
    # MFCC is a matrix of dimension Tl x N where Tl = length of all
    # utterances combined
    t = len(mfcc)
    n = len(mfcc[0])
    covariance = []
    for i in range(0, n):      # 0 - 25
        sum = 0
        for j in range(0, t):  # 0 - T
            sum += (mfcc[j][i] - mean[i]) ** 2
        covariance.append(sum / t)
    return covariance


# Equation from lecture 9, F9/77
# passed_observation:  Probability of the MFCC matrix given a state
# passed_trans:        Input transition matrix
# passed_mfcc:         Input set of mfcc vectors
# states:              Number of states in the HMM
# t:                   Location in 1:T in the MFCC matrix
# def get_alphas(passed_observation, passed_trans, passed_mfcc, states, t):
#     if t < 1:
#         # Reached beginning of the HMM.  Need to end the recursion
#         # At the first time step, the algorithm will initialize to
#         # the first state meaning q1 = 1 and all other q's = 0.
#         alpha_initial = [0] * len(passed_trans[0])
#         alpha_initial[0] = 1
#         print(alpha_initial)
#         return [alpha_initial]
#     last_alpha = get_alphas(passed_observation,
#                             passed_trans,
#                             passed_mfcc,
#                             states,
#                             t - 1)
#     current_alpha = []
#     for i in range(0, states):
#         sum = 0
#         for j in range(0, states):
#             # i = TO state, j = FROM state
#             transition_probability = passed_trans[j][i]
#             sum += transition_probability * last_alpha[t - 1][j]
#         observation = passed_observation[t - 1][i]
#         current_alpha.append(observation * sum)
#     last_alpha.append(current_alpha)
#     print(last_alpha)
#     return last_alpha


def get_alphas(passed_mean, passed_covariance, passed_trans, passed_mfcc, states, t):
    if t < 1:
        # Reached beginning of the HMM.  Need to end the recursion
        # At the first time step, the algorithm will initialize to
        # the first state meaning q1 = 1 and all other q's = 0.
        alpha_initial = [0] * len(passed_trans[0])
        alpha_initial[0] = 1
        return [alpha_initial]
    last_alpha = get_alphas(passed_mean,
                            passed_covariance,
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
            sum += transition_probability * last_alpha[t - 1][j]
        noise = np.random.normal(0, .01, len(passed_mean))
        observation = prob_observation_v2(passed_mean + noise,
                                          passed_covariance,
                                          passed_mfcc[t - 1])
        current_alpha.append(observation * sum)
    last_alpha.append(current_alpha)
    return last_alpha


# Equation from lecture 9, F14/77
# def get_betas(passed_observation, passed_covar,
#               passed_trans, passed_mfcc, states, t):
#     if t > len(passed_mfcc) - 1:
#         last_beta = [1] * len(passed_trans)
#         return [last_beta]
#     next_beta = get_betas(passed_observation, passed_covar,
#                           passed_trans, passed_mfcc, states, t + 1)
#     current_beta = []
#     for i in range(0, states):
#         sum = 0
#         for j in range(0, states):
#             # i = FROM state, j = TO state
#             transition_probability = passed_trans[i][j]
#             sum += passed_covar[j] * transition_probability * next_beta[j][0]
#         current_beta.append(sum)
#     return current_beta.extend(next_beta)


# Equation from lecture 9, F14/77
def get_betas(passed_covar, passed_trans, passed_mfcc, states, t):
    if t > len(passed_mfcc) - 1:
        last_beta = [1] * len(passed_trans)
        return [last_beta]
    next_beta = get_betas(passed_covar, passed_trans,
                          passed_mfcc, states, t + 1)
    current_beta = []
    for i in range(0, states):
        sum = 0
        for j in range(0, states):
            # i = FROM state, j
            # = TO state
            transition_probability = passed_trans[i][j]
            sum += passed_covar[j] * transition_probability * next_beta[0][j]
        current_beta.append(sum)
    current_beta = [current_beta]
    current_beta.extend(next_beta)
    return current_beta


# Equation from lecture 9, F19/77
# alpha:       Alpha matrix
# beta:        Beta matrix
# num_states:  Number of states
# t:           Time step
def get_gamma(alpha, beta, num_states, t):
    a = alpha[t]
    b = beta[t]
    gamma = []
    sum = 0
    for i in range(0, num_states):
        sum += a[i] * b[i]
    for i in range(0, num_states):
        tmp = a[i] * b[i] / sum
        gamma.append(tmp)
    return gamma
# returns a 1xQ vector


# def prob_observation_v2(mean, covariance, mfcc)
# Equation in lecture 9, F21/77
# alpha:       QxT Alpha matrix where Q = # of states
# beta:        QxT Beta matrix
# transition:  QxQ Transition matrix
# covariance:  NxN Covariance matrix
# mfcc:        TxN MFCC matrix
# mean:        1xN Mean vector where N = number of MFCC coefficients
# states:      Number of states in HMM
# t:           Time step in the MFCC
def get_xi(alpha, beta, transition, covariance, mfcc, mean, states, t):
    # observation
    mfcc_v = mfcc[t]
    p_o = prob_observation_v2(mean, covariance, mfcc_v)   # 1x1
    a = alpha(t - 1)  # Qx1
    b = beta(t)  # Qx1
    e = prob_evidence(alpha, beta)[t]
    xi_matrix = []
    for i in range(0, states):
        tmpa = a[i]
        tmpb = b[i]
        xi_vector = []
        for j in range(0, states):
            p_transition = transition[i][j]
            xi_vector.append((p_o * p_transition * tmpa * tmpb) / e)
        xi_matrix.append(xi_vector)
    return xi_matrix


def initial_state(alpha, beta, states):
    a = alpha[0]
    b = beta[0]
    sum = np.dot(a, b)
    init = []
    for i in range(0, states):
        init.append(a[i] * b[i] / sum)
    return init


def get_mfcc(waveform, hamming, fs):
    framerate = 100
    maxC = 13
    delta_diameter = 2
    mfcc = produce_mfcc(waveform, hamming, fs, framerate, maxC - 1)
    delta = get_delta(mfcc, delta_diameter)
    tmp = mfcc[delta_diameter : len(mfcc) - delta_diameter]
    features = []
    for i in range(0, len(delta)):
        features.append(np.concatenate((tmp[i], delta[i])))
    return features


def combine_mfcc(repeats, name, window, f):
    mfcc = []
    repeats = 1
    for i in range(0, repeats):
        # filename = name + "_%i.wav" % i
        filename = name + "_1.wav"
        data = wv.read(filename)[1]
        mfcc.extend(get_mfcc(data, window, f))
    return mfcc


def learn_initial(alphas, betas):
    for ell in range(0, len(alphas)):
        currenta = alphas[ell][0]
        currentb = betas[ell][0]
        sum = 0
        for q in range(0, len(currenta)):
            sum += currenta * currentb
        result = np.multiply(currenta, currentb)
        result = result / len(alphas)
        return result


# Equation 9.54 in lecture 9, F53/77
# alphas   Q x T x \ell Set of alpha matrices
# betas    Q x T x \ell Set of beta matrices
# mfccs    Set of MFCC's
# states   Number of states
def learn_transition(alphas, betas, mfccs, states):
    sum_transition = [[0] * states] * states
    for ell in range(0, len(alphas)):
        current_mfcc = mfccs[ell]
        sumg = [0] * states
        sumx = [[0] * states] * states
        for t in range(1, len(alphas[0])):
            currentg = get_gamma(
                alphas[ell],
                betas[ell],
                len(alphas[0][0][0]),
                t)
            currentx = get_xi(
                alphas[ell],
                betas[ell],
                odessa_transition_guess,
                global_covariance,
                current_mfcc,
                global_mean,
                states,
                t)
            sumg = np.add(sumg, currentg)
            sumx = np.add(sumx, currentx)
        sum_transition += np.divide(sumx, sumg)
    return sum_transition / len(alphas)


# =============#
#     START    #
# -------------#
fs = 16000
sd.default.samplerate = fs
sd.default.channels = 1
window = 25
hamming_window = make_window(25, fs)
utterances = 1

# ===========
# EM Learning
# ===========
# Concatenating all MFCCs from ALL utterances of everything
global_mfcc = combine_mfcc(utterances, "Odessa", hamming_window, fs)
global_mfcc.extend(combine_mfcc(utterances, "LightsOn", hamming_window, fs))
global_mfcc.extend(combine_mfcc(utterances, "LightsOff", hamming_window, fs))
global_mfcc.extend(combine_mfcc(utterances, "PlayMusic", hamming_window, fs))
global_mfcc.extend(combine_mfcc(utterances, "StopMusic", hamming_window, fs))
global_mfcc.extend(combine_mfcc(utterances, "Time", hamming_window, fs))

global_mean = get_mean(global_mfcc)
global_covariance = get_covariance(global_mfcc, global_mean)

# Odessa Training
odessa_states = 10
odessa_alphas = [None] * utterances
odessa_betas = [None] * utterances
odessa_transition_guess = get_transition(odessa_states)
odessa_training_mfcc = []
for i in range(0, utterances + 1):
    filename = "Odessa_%i.wav" % i
    data = wv.read(filename)[1]
    odessa_training_mfcc.append(get_mfcc(data, hamming_window, fs))
for i in range(0, utterances):
    filename = "Odessa_%i.wav" % (i + 1)
    data = wv.read(filename)[1]
    tmp = get_mfcc(data, hamming_window, fs)
    # odessa_alpha[i] = get_alphas(
    #     prob_observation(global_mean, global_covariance, tmp, odessa_states),
    #     odessa_transition,
    #     tmp,
    #     odessa_states,
    #     len(tmp))
    odessa_alphas[i] = get_alphas(
        global_mean,
        global_covariance,
        odessa_transition_guess,
        tmp,
        odessa_states,
        len(tmp))
    print(odessa_alphas[0])
    odessa_betas[i] = get_betas(
        global_covariance,
        odessa_transition_guess,
        tmp,
        odessa_states,
        0)
    print(odessa_betas[0])
odessa_initial_em = learn_initial(
    odessa_alphas, 
    odessa_betas)
odessa_transition_em = learn_transition(
    odessa_alphas,
    odessa_betas,
    odessa_training_mfcc,
    odessa_states)
odessa_gaussians_em = learn_gaussians()

def learn_gaussians(alphas, betas, states, mfccs):
    sum_num = 0
    sum_den = 0
    for ell in range(0, len(alphas)):
        a = alphas[ell]
        b = betas[ell]
        mfcc = mfccs[ell]
        for t in range(0, len(alphas[0])):
            # MEAN
            # pseudo code:
            gamma_t = get_gamma(a, b, states, t)
            sum_num += np.multiply(mfcc[t], gamma_t)
            sum_den += gamma_t
        for t in range(0, len(alphas[0])):
            # COVARIANCE
            # pseudo code:
            
            # ((x(t) - mean) ^ 2) * gamma_q(t)
            # divided by
            # gamma_q(t)

def lean_covariances(alphas, betas, mfccs, states, learned_mean):
    # Should end up with a 3D matrix of size N x N x Q 
    # i.e. Each state should have an N x N covariance matrix
    for ell in range(0, len(alphas)):
        mfcc = mfccs[ell]
        a = alphas[ell]
        b = betas[ell]
        for t in range(0, len(alphas[0])):
            gamma_t = get_gamma(a, b, states, t)
            # pseud code:
            for q in range(0, states): 
                # ((mfcc[t] - mean[q]) ^ 2) * gamma_t[q]
                gamma
            
    # YOU ARE HERE
    # YOU ARE HERE
    # YOU ARE HERE
    # YOU ARE HERE
    # YOU ARE HERE
    # YOU ARE HERE
    
    
    
# Equation from lecture 9, F19/77
# alpha:       Alpha matrix
# beta:        Beta matrix
# num_states:  Number of states
# t:           Time step
# def get_gamma(alpha, beta, num_states, t):





#     # def get_alphas(passed_observation, passed_trans, passed_mfcc, states, t):
#     alpha = get_alphas(mfcc, odessa_trans)
#     beta = get_betas(mfcc,)
#     pi_state = initial_state(alpha, beta, )
#     mfcc = [get_mfcc(data, hamming_window, fs)]
#     mfcc_mean = get_mean(mfcc)
#     covariance = get_covariance(mfcc, mfcc_mean)
#
# odessa_states = 10
# odessa_mean = [1 / odessa_states] * odessa_states
# odessa_trans = get_transition(odessa_states)
# mfcc_observation = prob_observation(odessa_mean, covariance, mfcc, odessa_states)
#
# alpha_t = len(mfcc)
# odessa_alphas = get_alphas(
#     mfcc_observation, odessa_trans, mfcc,
#     odessa_states, alpha_t)
# odessa_betas = get_betas(
#     mfcc_observation, covariance, odessa_trans,
#     mfcc, odessa_states, 0)
# odessa_gamma = get_gamma(
#     odessa_alphas, odessa_betas, odessa_states)
# odessa_xi = get_xi(
#     odessa_alphas, odessa_betas, mfcc_observation,
#     odessa_trans, covariance, mfcc, odessa_states)
#
# odessa_utterances = 20
# hamming_window = make_window(window, fs)



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





