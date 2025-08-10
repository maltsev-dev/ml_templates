import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_waveform(waveform, title="Waveform"):
    plt.figure()
    plt.plot(waveform.numpy())
    plt.title(title)
    plt.show()

def plot_spectrogram(spectrogram, title="Spectrogram"):
    log_spec = tf.math.log(spectrogram + 1e-6)
    plt.figure(figsize=(10, 4))
    plt.imshow(tf.transpose(log_spec).numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()
