# ====================== Data preparation ======================
# ====================== Base model ======================

# Build model: feature extractor + classifier
model = tf.keras.Sequential([
    feature_extractor,                                      # Model extracts features from image
    tf.keras.layers.Dense(num_classes, activation='softmax') # Final classification layer (2 classes — cat/dog)
])

# ====================== QAT Optimization Start======================

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

# ====================== QAT Optimization End======================

# Number of epochs (full passes through training dataset)
EPOCHS = 5

# Train model:
# - train_batches — training data
# - validation_batches — data for evaluation after each epoch
hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)

# ====================== Save SavedModel ======================
# ====================== Model conversion without optimization======================

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

# ====================== Interpreter initialization ======================
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)
print('Quant TF test accuracy:', q_aware_model_accuracy)