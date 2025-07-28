# ====================== Data preparation ======================
# ====================== Base model ======================
# ====================== Save SavedModel ======================
# ====================== Model conversion with quantization======================

import pathlib

# Load model into TFLiteConverter from previously saved directory
converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)

# Representative function - TFLite will use these examples
# to estimate the range of input values and properly quantize the model.
def representative_data_gen():
    for input_value, _ in test_batches.take(100):
        yield [input_value]                         # important: return a list with a tensor (the input can be a tuple/list)

# Specify the function for generating representative data for post-training quantization
converter.representative_dataset = representative_data_gen

# Restrict operations to INT8 only (everything will be quantized)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Additionally, you can specify that inputs/outputs should also be int8 (not float32)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

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