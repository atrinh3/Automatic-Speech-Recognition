# from .user import HiddenMarkovModel as hmm

import AlphaBeta
import Posteriors as post
import HiddenMarkovModel as hmm
import EMTraining as emt
import MakeMFCC
import sounddevice as sd
import scipy.io.wavfile as wv

fs = 16000
sd.default.samplerate = fs
sd.default.channels = 1
window = 25
hamming_window = MakeMFCC.make_window(25, fs)
utterances = 20

odessa_hmm = hmm.HiddenMarkovModel("Odessa", 10)
lights_on_hmm = hmm.HiddenMarkovModel("Lights On", 20)
lights_off_hmm = hmm.HiddenMarkovModel("Lights off", 20)
play_music_hmm = hmm.HiddenMarkovModel("Play Music", 15)
stop_music_hmm = hmm.HiddenMarkovModel("Stop Music", 20)
time_hmm = hmm.HiddenMarkovModel("Time", 20)

odessa_mfccs = []
lights_on_mfccs = []
lights_off_mfccs = []
play_music_mfccs = []
stop_music_mfccs = []
time_mfccs = []
for i in range(1, utterances + 1):
    filename = "new_Odessa_%i.wav" % i
    data = wv.read(filename)[1]
    odessa_mfccs.append(MakeMFCC.get_mfcc(data, hamming_window, fs))
    filename = "new_LightsOn_%i.wav" % i
    data = wv.read(filename)[1]
    lights_on_mfccs.append(MakeMFCC.get_mfcc(data, hamming_window, fs))
    filename = "new_LightsOff_%i.wav" % i
    data = wv.read(filename)[1]
    lights_off_mfccs.append(MakeMFCC.get_mfcc(data, hamming_window, fs))
    filename = "new_PlayMusic_%i.wav" % i
    data = wv.read(filename)[1]
    play_music_mfccs.append(MakeMFCC.get_mfcc(data, hamming_window, fs))
    filename = "new_StopMusic_%i.wav" % i
    data = wv.read(filename)[1]
    stop_music_mfccs.append(MakeMFCC.get_mfcc(data, hamming_window, fs))
    filename = "new_Time_%i.wav" % i
    data = wv.read(filename)[1]
    time_mfccs.append(MakeMFCC.get_mfcc(data, hamming_window, fs))

# Train "Odessa"

# Problem 3(e).
# Delta features (20 points)
# Once you have your matrix, compute delta features. Use a window size of M=2.
# Concatenate your feature matrix with your delta-feature matrix to create
# a feature matrix of dimension 26 x(T - 2M).
delta = get_delta(mfcc, delta_diameter)
tmp = mfcc[delta_diameter:len(mfcc)-delta_diameter]
features = []

for i in range(0, len(delta)):
    a1 = tmp[i]
    a2 = delta[i]
    test = np.concatenate((a1, a2))
    features.append(test)


# get Odessa MFCCs
# get Odessa alphas
# get Odessa betas
# train initial state
emt.train_initial_state(
    odessa_alphas, odessa_betas, odessa_initial_guess, odessa_mean,
    odessa_covariance, odessa_transition, odessa_mfccs)
# train transition matrix
emt.train_transition(
    odessa_alphas, odessa_betas, odessa_mfccs, odessa_mean, odessa_covariance,
    odessa_transition)
# train gaussian
emt.train_gaussian(
    odessa_alphas, odessa_betas, odessa_mfccs, odessa_mean, odessa_covariance,
    odessa_transistion)


# Train "Turn on the lights"
emt.train_initial_state(
    lights_on_alphas, lights_on_betas, lights_on_initial_guess, lights_on_mean,
    lights_on_covariance, lights_on_transition, lights_on_mfccs)
emt.train_transition(
    lights_on_alphas, lights_on_betas, lights_on_mfccs, lights_on_mean,
    lights_on_covariance, lights_on_transition)
emt.train_gaussian(
    lights_on_alphas, lights_on_betas, lights_on_mfccs, lights_on_mean,
    lights_on_covariance, lights_on_transition)


# Train "Turn off the lights"
emt.train_initial_state(
    lights_off_alphas, lights_off_betas, lights_off_initial_guess,
    lights_off_mean, lights_off_covariance, lights_off_transition,
    lights_off_mfccs)
emt.train_transition(
    lights_off_alphas, lights_off_betas, lights_off_mfccs, lights_off_mean,
    lights_off_covariance, lights_off_transition)
emt.train_gaussian(
    lights_off_alphas, lights_off_betas, lights_off_mfccs, lights_off_mean,
    lights_off_covariance, lights_off_transition)


# Train "Play music"
emt.train_initial_state(
    play_music_alphas, play_music_betas, play_music_initial_guess,
    play_music_mean, play_music_covariance, play_music_transition,
    play_music_mfccs)
emt.train_transition(
    play_music_alphas, play_music_betas, play_music_mfccs, play_music_mean,
    play_music_covariance, play_music_transition)
emt.train_gaussian(
    play_music_alphas, play_music_betas, play_music_mfccs, play_music_mean,
    play_music_covariance, play_music_transition)


# Train "Stop music"
emt.train_initial_state(
    stop_music_alphas, stop_music_betas, stop_music_initial_guess,
    stop_music_mean, stop_music_covariance, stop_music_transition,
    stop_music_mfccs)
emt.train_transition(
    stop_music_alphas, stop_music_betas, stop_music_mfccs, stop_music_mean,
    stop_music_covariance, stop_music_transition)
emt.train_gaussian(
    stop_music_alphas, stop_music_betas, stop_music_mfccs, stop_music_mean,
    stop_music_covariance, stop_music_transition)


# Train "What time is it"
emt.train_initial_state(
    time_alphas, time_betas, time_initial_guess, time_mean, time_covariance,
    time_transition, time_mfccs)
emt.train_transition(
    time_alphas, time_betas, time_mfccs, time_mean, time_covariance,
    time_transition)
emt.train_gaussian(
    time_alphas, time_betas, time_mfccs, time_mean, time_covariance,
    time_transition)

