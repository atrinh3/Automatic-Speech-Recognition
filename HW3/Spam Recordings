import sounddevice as sd
import scipy.io.wavfile as wv


def save_wav(filename, recording, fs):
    print("Recording saved as \"" + filename + "\".")
    wv.write(filename, fs, recording)


def make_recording(freq, time, name, id, r):
    print("Recording %s %i out of %i..." % (name, id+1, r))
    s = sd.rec(int(time * fs),
               blocking=True,
               dtype='int16')
    print("Stopped recording.")
    print("Playing back recording %s %i out of %i..." % (name, id+1, r))
    sd.play(s, blocking=True)
    print("Playback finished.")
    # save as .wav file
    save_wav(name, s, freq)


# =============#
#    START     #
# -------------#

CHANNELS = 1
fs = 16000
repeats = 10
sd.default.samplerate = fs
sd.default.channels = CHANNELS
start = 11

# Odessa
duration = 2  # seconds
# start = 11
for i in range(0, repeats):
    filename = "Odessa_%i.wav" % (i + start)
    make_recording(fs, duration, filename, i, repeats)

# Turn on the lights
duration = 3  # seconds
# start = 11
for i in range(0, repeats):
    filename = "LightsOn_%i.wav" % (i + start)
    make_recording(fs, duration, filename, i, repeats)

# Turn off the lights
duration = 3  # seconds
# start = 11
for i in range(0, repeats):
    filename = "LightsOff_%i.wav" % (i + start)
    make_recording(fs, duration, filename, i, repeats)

# What time is it
duration = 3  # seconds
# start = 11
for i in range(0, repeats):
    filename = "Time_%i.wav" % (i + start)
    make_recording(fs, duration, filename, i, repeats)

# Play music
duration = 2  # seconds
# start = 11
for i in range(0, repeats):
    filename = "PlayMusic_%i.wav" % (i + start)
    make_recording(fs, duration, filename, i, repeats)

# Stop music
duration = 2  # seconds
# start = 11
for i in range(0, repeats):
    filename = "StopMusic_%i.wav" % (i + start)
    make_recording(fs, duration, filename, i, repeats)
