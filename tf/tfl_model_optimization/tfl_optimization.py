# ====================== Data preparation ======================
# ====================== Base model ======================
# ====================== Save SavedModel ======================
# ====================== Model conversion with optimization======================

import pathlib

# Load model into TFLiteConverter from previously saved directory
converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model to TFLite format
tflite_model = converter.convert()
# Specify directory to save the model
tflite_models_dir = pathlib.Path("/tmp/")  # temporary directory (can change path)

# Define path to model file
tflite_model_file = tflite_models_dir / 'model1.tflite'
# Save byte representation of model to file
tflite_model_file.write_bytes(tflite_model)

# ====================== Interpreter initialization ======================
# ====================== Utility functions for plotting ======================