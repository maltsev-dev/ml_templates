import tensorflow as tf
import os
import pathlib

AUTOTUNE = tf.data.AUTOTUNE
DATASET_PATH = "data/mini_speech_commands"

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def prepare_dataset():
    data_dir = pathlib.Path(DATASET_PATH)
    file_paths = list(data_dir.glob('*/*.wav'))
    file_paths = [str(path) for path in file_paths]
    files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    waveforms_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    return waveforms_ds

if __name__ == "__main__":
    ds = prepare_dataset()
    for waveform, label in ds.take(1):
        print(label.numpy())
