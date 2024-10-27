
import librosa
import numpy as np
from matplotlib import pyplot as plt
# 3.创建一个独立的画图文件，方便调用
    # 时域
def plot_waveform(waveform, sr, title = "Waveform"):
    waveform = np.array(waveform)
    samples = waveform.size
    plt.figure(figsize = (20, 10))
    time_scale = np.linspace(0, samples/sr, num = samples)
    plt.plot(time_scale, waveform, linewidth = 1)
    plt.title(title)
    plt.grid(True)
    plt.show()

    # 频域
def plot_waveform_fft(waveform, sr, n_fft, title = "Waveform_FT"):
    waveform = np.array(waveform)
    waveform_fft = np.fft.rfft(waveform, n_fft)
    freq_scale = np.linspace(0, sr / 2, num = int(n_fft / 2) + 1)
    plt.figure(figsize = (20, 10))
    plt.plot(freq_scale, waveform_fft, linewidth = 1)
    plt.title(title)
    plt.grid(True)
    plt.show()

    # 二维
def plot_spectrogram(spectrogram, title = "Spectrogram(dB)"):
    plt.imshow(librosa.power_to_db(spectrogram), )
    plt.title(title)
    plt.xlabel("Frame/s")
    plt.ylabel("Frequency/Hz")
    plt.colorbar()
    plt.show()
    