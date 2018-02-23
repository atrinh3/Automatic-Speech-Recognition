import math
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt


def get_sign(data):
    if data >= 0:
        return 1
    return -1


def find_zeros(data):
    length = len(data)
    # print(length)
    sum = 0
    for i in range(1, window):
        # print(data[i])
        # print(data[i-1])
        a = get_sign(data[i])
        b = get_sign(data[i-1])
        sum += int(abs(a-b)/2)
    return sum
    # return sum/length


def get_energy(data, window):
    e = []
    i = 0
    # print(len(data))
    while i+window <= len(data):
        sum = 0
        for j in range(0, int(window)):
            sum += abs(data[int(i+j)])
        e.append(sum)
        i += window
    return e


def listen(data, energy, lower, upper, frames, width, std):
    length = len(energy)
    tmp = None
    for i in range(0, length):
        if energy[i] < lower:
            tmp = None
        if energy[i] >= lower:
            tmp = i
        if energy[i] >= upper:
            valid = zero_check(data, tmp, frames, width, std)
            if valid is not None:
                return [tmp, valid]
    return [tmp, None]


def zero_check(data, N1, frames, width, zt):
    count = 0
    least = N1*width
    for i in range(1, frames):
        id = N1*width - i*width
        start = int(id)
        end = int(start + width)
        if find_zeros(data[start:end]) >= math.floor(zt):
            count += 1
            if id < least:
                least = id
    if count >= 3:
        return int(least / width)
    return None


# Rabiner/Sambur Algorithm
fs = 16000
window = 10  # in ms
filename = 'StopMusic_1.wav'
signal = wv.read(filename)[1]

sample = fs * window / 1000  # qty of data points in windowed sample

# 1) Compute Es(n) for n ranging over the segment of audio.
energy = get_energy(signal, sample)

# 2) Compute IMX and IMN, the max and min respectively of Es(n).
imn = energy[0]
imx = imn
for i in range(0, len(energy)):
    if energy[i] < imn:
        imn = energy[i]
    elif energy[i] > imx:
        imx = energy[i]

# 3) Set ITL = min(0.03 * (IMX - IMN) + IMN, 4 * IMN), and
#        ITU = 5 * ITL,
#        the upper and lower thresholds of the engery respectively
itl = min(0.03 * (imx - imn) + imn, 4 * imn)
itu = 5 * itl

# 4) IZCT = min(IF, IZC+ 2oIZC) where IF = 25, and IZC(resp. oIZC)
#    is the mean (resp. standard deviation) of the zero crossing array
#    during silence (i.e., the first 100ms of the signal). IZCT is the
#    zero crossing threshold.
assumed_silence = 100  # assume first 100ms is always silent
zc_control = 25
zc_array = []
for i in range(0, int(assumed_silence / window)):
    start = int(i * sample)
    end = int(start + sample)
    zc_array.append(find_zeros(signal[start:end]))
zc_mean = float(np.mean(zc_array))
std = 0
for i in zc_array:
    std += (i - zc_mean) * (i - zc_mean)
std = std / len(zc_array)
std = math.sqrt(std)
izct = min([zc_control, zc_mean + 2*std])
print('mean is %f, std is %f, izct is %f' % (zc_mean, std, izct))

# 5) Start at the beginning of the speech signal. Search towards the
#    center until the first frame where Es(n) goes above ITL and then
#    above ITU without first going below ITL. Call this point N1.


# 6) Search starting at frame N1 to frame N1 - 25 (in this case a
#    250ms interval) and if the number of frames whose zero-crossing
#    is greater than IZCT is three or more, the starting point it
#    changed to the first such frame N1' in that 25 frame interval
#    whose threshole exceeds IZCT. Otherwise leave the segment point
#    alone.

intervals = 25  # frames
[N1, N_Prime] = listen(signal, energy, itl, itu, intervals, sample, std)
print("N1 is %f and N1' is %f" % (N1, N_Prime))
