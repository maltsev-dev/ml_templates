import tensorflow as tf
import os

MODEL_PATH = "models/model.h5"
TFLITE_MODEL_PATH = "models/model.tflite"

def convert_to_tflite():
    if not os.path.exists(MODEL_PATH):
        print("Trainde model not found! Please train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved at: {TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    convert_to_tflite()
