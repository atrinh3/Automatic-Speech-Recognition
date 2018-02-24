import sounddevice as sd
import soundfile as sf
import numpy as np
import math
import sys


def get_sign(data):
    if data >= 0:
        return 1
    return -1


def find_zeros(data):
    sum = 0
    for i in range(1, len(data)):
        a = get_sign(data[i])
        b = get_sign(data[i - 1])
        sum += int(abs(a - b) / 2)
    return sum


def get_energy(data, window):
    e = []
    i = 0
    while i + window <= len(data):
        sum = 0
        for j in range(0, int(window)):
            sum += abs(data[int(i + j)])
        e.append(sum)
        i += window
    return e


def listen(data, energy, lower, upper, frames, width, zt):
    length = len(energy)
    tmp = None
    for i in range(0, length):
        if energy[i] < lower:
            tmp = None
        if energy[i] >= lower:
            tmp = i
        if energy[i] >= upper:
            valid = zero_check(data, tmp, frames, width, zt)
            if valid is not None:
                return True
    return False


def zero_check(data, N1, frames, width, zt):
    count = 0
    least = N1 * width
    for i in range(1, frames):
        id = N1 * width - i * width
        start = int(id)
        end = int(start + width)
        # print('index %i' % i)
        # print("start %i, to end %i" % (start, end))
        if find_zeros(data[start:end]) >= math.floor(zt):
            count += 1
            if id < least:
                least = id
    if count >= 3:
        return int(least / width)
    return None


def detect_voice(signal):
    # Rabiner/Sambur Algorithm
    fs = 16000
    window = 10  # in ms
    assumed_silence = 100  # assume first 100ms is always silent
    itl = 8081
    # itu = 40407
    itu = 30000

    sample = fs * window / 1000
    energy = get_energy(signal, sample)

    # 2) Compute IMX and IMN.
    imn = energy[0]
    imx = imn
    for i in range(0, len(energy)):
        if energy[i] < imn:
            imn = energy[i]
        elif energy[i] > imx:
            imx = energy[i]

    # 3) Set ITL and ITU
    itl = min(0.03 * (imx - imn) + imn, 4 * imn)
    # itl = 0.03 * (imx - imn) + imn
    itu = 5 * itl

    # 4) Find mean and std for zero crossings
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
    izct = min([zc_control, zc_mean + 2 * std])

    # 5 & 6) Find N1 & N'
    intervals = 25  # frames
    return listen(signal, energy, itl, itu, intervals, sample, izct)


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if detect_voice(indata):
        print("Voice detected")
    else:
        print("No voice")
    # if status:
    #     print(status)
    #     print(sys.stderr)
    wavfile.write(indata)


# =============#
#     START    #
# -------------#
chunk = 4096
duration = 30
channels = 1
fs = 16000
filename = "Problem4Stream.wav"
sd.default.samplerate = fs
sd.default.channels = channels

wavfile = sf.SoundFile(filename,
                       mode='wb',
                       samplerate=fs,
                       channels=channels,
                       subtype='PCM_16')

with sd.InputStream(callback=callback, blocksize=chunk):
    sd.sleep(int(duration * 1000))
