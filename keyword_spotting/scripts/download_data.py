import os
import tensorflow as tf

DATASET_PATH = "data/mini_speech_commands"

def download():
    if not os.path.exists(DATASET_PATH):
        print("Downloading...")
        dataset_url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
        tf.keras.utils.get_file(
            fname="mini_speech_commands.zip",
            origin=dataset_url,
            extract=True,
            cache_dir='.',
            cache_subdir='data'
        )
    else:
        print("Successful completed")

if __name__ == "__main__":
    download()
