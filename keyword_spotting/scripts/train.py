import os
import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
DATASET_PATH = "data/mini_speech_commands"
LABELS = ['left', 'right', 'stop', 'go']
MODEL_DIR = "models/"

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

def create_model(input_shape, num_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_labels)
    ])
    return model

def main():
    data_dir = pathlib.Path(DATASET_PATH)
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
    train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=SEED)

    train_ds = preprocess_dataset(train_files)
    val_ds = preprocess_dataset(val_files)

    train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape[1:]

    model = create_model(input_shape, len(LABELS))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    checkpoint_path = os.path.join(MODEL_DIR, "model.h5")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=3)
        ]
    )

    print(f"Model saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
