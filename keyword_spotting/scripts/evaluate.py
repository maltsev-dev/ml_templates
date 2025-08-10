import os
import pathlib
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/mini_speech_commands"
LABELS = ['left', 'right', 'stop', 'go']
MODEL_PATH = "models/model.h5"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

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

def get_spectrogram(waveform):
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def get_spectrogram_and_label_id(waveform, label):
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(tf.cast(tf.equal(LABELS, label), tf.int32))
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds

def main():
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found!")
        return

    print("Accuracy evaluation...")

    data_dir = pathlib.Path(DATASET_PATH)
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
    _, val_files = train_test_split(filenames, test_size=0.2, random_state=42)

    val_ds = preprocess_dataset(val_files)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(val_ds)
    print(f"\n🎯 Accuracy at validation: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
