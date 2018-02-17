import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wv
import sounddevice as sd


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


def save_wav(filename, recording, fs):
    print("Recording saved as \"" + filename + "\".")
    wv.write(filename, fs, recording)


def make_spectrogram(s, nfft, npower, fs, window):
    # Now produce a spectrogram.
    # First, set the window size, which is the number of samples
    # to process at a time over which each vertical column of the
    # spectrum corresponds to. The python specgram calls this NFFT for
    # some reason, but we’re going to name a parameter window_size, and this
    # is going to be in units of milliseconds, and we’ll convert it
    # to the corresponding ’nfft’ value.
    next_power = npower

    my_pad = int(np.power(2, (next_power - 1) + np.ceil(np.log2(nfft))))

    # Note that my_pad is a power of 2, and this is what the final FFT length
    # python will use.

    # Next, we set how many points of the speech windows overlap between successive
    # windows. We’re going to say that 7/8’ths of the points overlap (so stepping
    # by 1/8th of a window at each column of the spectrogram. Note that
    # how many points this is depends on the window size.
    my_noverlap = int(float(nfft) * 7.0 / 8.0)

    # Lastly, we’re going to select a color map. ’jet’ has commonly been used
    # in matlab, but there is some concern about using ’jet’ as a color map as
    # the color intensity (or luminosity) is dark for both low and high magnitude
    # values (so it doesn’t plot well when plotted in B&W). Hence, we’re going
    # to use the colormap called ’cubehelix’, although if you change this to
    # use ’jet’, you’ll see results that probably look more familiar since ’jet’
    # is so widely used when plotting in color.
    my_cmap = plt.get_cmap('cubehelix')

    # Lastly, plot the spectrogram.
    fig, ax = plt.subplots()
    ax.set_title(
        'A Spectrogram w/ Window Size: %i ms & FFT Size: %i'
        % (window, my_pad))
    ax.set_xlabel('time(seconds)')
    ax.set_ylabel('frequency')

    Pxx, freqs, bins, im = plt.specgram(
        s[1],
        NFFT=nfft,
        Fs=fs,
        pad_to=my_pad,
        noverlap=my_noverlap,
        cmap=my_cmap)
    fig.colorbar(im).set_label('Intensity(dB)')
    plt.show()


def prob_1_a(w, data, fs, row):
    fft = int(float(fs) * float(w) / 1000.0)
    for i in range(1, 5):
        make_spectrogram(data, fft, i, fs, w)


# =============#
#    START    #
# -------------#
fs = 16000  # 16kHz sample rate
duration = 4  # 4 seconds
filename = "prob1a.wav"
sd.default.samplerate = fs
sd.default.channels = 1

# make_recording(fs, duration, filename)
signal = wv.read(filename)

# Prob 1a

prob_1_a(5, signal, fs, 1)
prob_1_a(10, signal, fs, 2)
prob_1_a(25, signal, fs, 3)
prob_1_a(50, signal, fs, 4)
prob_1_a(100, signal, fs, 5)

# window size 5 = 80 samples

# 128
# 256
# 512
# 1024
# 2048
# 4096
# 8192
# 16384
