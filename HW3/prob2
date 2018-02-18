import sounddevice as sd
import scipy.io.wavfile as wv
import numpy as np
import matplotlib.pyplot as plt


def save_wav(filename, recording, fs):
    print("Recording saved as \"" + filename + "\".")
    wv.write(filename, fs, recording)


def make_recording(freq, time, name):
    print("Recording...")
    s = sd.rec(int(time * fs),
               blocking=True,
               dtype='int16')
    print("Stopped recording.")
    print("Playing back recording...")
    sd.play(s, blocking=True)
    print("Playback finished.")
    # save as .wav file
    save_wav(name, s, freq)


def make_spectrogram(s, nfft, fs, window):
    next_power = 1
    my_pad = int(np.power(2, (next_power - 1) + np.ceil(np.log2(nfft))))
    my_noverlap = int(float(nfft) * 7.0 / 8.0)
    my_cmap = plt.get_cmap('cubehelix')

    fig, ax = plt.subplots()
    # ax.set_title(
    #     'A Spectrogram w/ Window Size: %i ms & FFT Size: %i'
    #     % (window, my_pad))
    ax.set_xlabel('time(seconds)')
    ax.set_ylabel('frequency')
    # print(s.shape)
    # print(s[:,0])
    # print(len(s[1]))
    Pxx, freqs, bins, im = plt.specgram(
        # s[1],
        s,
        NFFT=nfft,
        Fs=fs,
        pad_to=my_pad,
        noverlap=my_noverlap,
        cmap=my_cmap)
    fig.colorbar(im).set_label('Intensity(dB)')
    # plt.show()


# =============#
#    START     #
# -------------#
duration = 4  # seconds
filename = "prob1a.wav"
CHANNELS = 1
fs = 16000
window_size = 25

sd.default.samplerate = fs
sd.default.channels = CHANNELS
# make_recording(fs, duration, filename)
s = wv.read(filename)
nfft = int(float(fs)*float(window_size)/1000.0)

tmp = s[1]
make_spectrogram(tmp, nfft, fs, window_size)
plt.title('Graph 1')
plt.show()

tmp = np.abs(s[1])
make_spectrogram(tmp, nfft, fs, window_size)
plt.title('Graph 2')
plt.show()

tmp = s[1]*(s[1] >= 0)
make_spectrogram(tmp, nfft, fs, window_size)
plt.title('Graph 3')
plt.show()

tmp = s[1]**2
make_spectrogram(tmp, nfft, fs, window_size)
plt.title('Graph 4')
plt.show()
