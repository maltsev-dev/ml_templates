# Specify required library versions
tf_version = "2.14.0"            # Desired TensorFlow version
hub_version = "0.15.0"           # Desired TensorFlow Hub version (community models)
datasets_version = "4.6.0"       # Desired TensorFlow Datasets version (ready datasets)
numpy_version = "1.26.4"         # Desired NumPy version (numerical arrays)
protobuf_version = "3.20.3"      # Desired protobuf version (serialization format used by TensorFlow)

# Import main libraries
import tensorflow as tf
import numpy as np

# Try to import additional modules
try:
    import tensorflow_hub as hub          # For loading models from https://tfhub.dev
    import tensorflow_datasets as tfds    # For using standard datasets
except:
    # If modules are missing — set variables to None to avoid errors later
    hub = None
    tfds = None

# Check if versions match the requirements
if (tf.__version__ != tf_version or
    (hub and hub.__version__ != hub_version) or
    (tfds and tfds.__version__ != datasets_version) or
    not np.__version__.startswith(numpy_version)):

    # Print current library versions that do not match
    print(f"Current TensorFlow version: {tf.__version__} → {tf_version}")
    if hub:
        print(f"Current TensorFlow Hub version: {hub.__version__} → {hub_version}")
    if tfds:
        print(f"Current TensorFlow Datasets version: {tfds.__version__} → {datasets_version}")
    print(f"Current NumPy version: {np.__version__} → {numpy_version}")

    # Uninstall current library versions (without confirmation -y)
    !pip uninstall -y tensorflow tensorflow_hub tensorflow_datasets numpy protobuf

    # Install required versions of all libraries
    !pip install tensorflow=={tf_version} \
                  tensorflow_hub=={hub_version} \
                  tensorflow_datasets=={datasets_version} \
                  numpy=={numpy_version} \
                  protobuf=={protobuf_version}

    # Notify user to restart the runtime
    print("\n Specified versions installed successfully.")
    print(" Please restart the runtime (Runtime > Restart session) and re-run the notebook.\n")
else:
    # If all versions match — do nothing
    print(" All packages are already at the specified versions.")


# ====================== Data preparation ======================

# Import required libraries
import numpy as np                         # Working with arrays
import matplotlib.pylab as plt             # Plotting
import tensorflow as tf                    # Main machine learning library
import tensorflow_hub as hub               # Loading pretrained models from TensorFlow Hub
import tensorflow_datasets as tfds         # Simplified work with ready datasets

# Override download URL for cats_vs_dogs dataset (sometimes needed for access)
setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

# Image formatting function:
# - Resizes to 224x224 (size expected by many CNNs, e.g., MobileNet)
# - Normalizes pixel values to [0, 1]
def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return  image, label

# Load "cats_vs_dogs" dataset from TensorFlow Datasets:
# - split into train (80%), validation (10%), and test (10%) parts
# - with_info=True allows getting metadata (e.g., number of classes)
# - as_supervised=True returns tuples (image, label), not dicts
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
# Get total number of examples and number of classes from metadata
num_examples = metadata.splits['train'].num_examples     # Number of images in the full dataset
num_classes = metadata.features['label'].num_classes     # Number of classes (2: cats and dogs)
print(num_examples)
print(num_classes)

# Set batch size
BATCH_SIZE = 32

# Create pipeline for training set:
# - shuffle 1/4 of examples
# - format images
# - batch
# - prefetch(1): loads data in advance (speeds up training)
train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)

# Similarly for validation, but without shuffling
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)

# For test, use batch size 1
test_batches = raw_test.map(format_image).batch(1)

# Take one batch from training data to check shape
for image_batch, label_batch in train_batches.take(1):
    pass

# Show batch shape (should be: (32, 224, 224, 3))
image_batch.shape

# ====================== Base model ======================

# Select pretrained model and its parameters:
module_selection = ("mobilenet_v2", 224, 1280)  # (model name, input size, output feature vector size)

handle_base, pixels, FV_SIZE = module_selection
# Compose link to model from TensorFlow Hub (feature vector extractor)
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)

# Input image size (width, height)
IMAGE_SIZE = (pixels, pixels)

# Print info about selected model
print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

# Load model from TensorFlow Hub as Keras layer
# - input_shape = (224, 224, 3) — input RGB image
# - output_shape = [1280] — output feature vector
# - trainable=False — freeze weights, do not train this layer (used as feature extractor)
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=False)

print("Building model with", MODULE_HANDLE)

# Build model: feature extractor + classifier
model = tf.keras.Sequential([
    feature_extractor,                                      # Model extracts features from image
    tf.keras.layers.Dense(num_classes, activation='softmax') # Final classification layer (2 classes — cat/dog)
])

# Show model architecture: number of layers, parameters
model.summary()

# Compile model:
# - optimizer='adam' — popular optimizer
# - loss='sparse_categorical_crossentropy' — loss for integer class labels (0 or 1)
# - metrics=['accuracy'] — show accuracy during training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Number of epochs (full passes through training dataset)
EPOCHS = 5

# Train model:
# - train_batches — training data
# - validation_batches — data for evaluation after each epoch
hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)

# ====================== Save SavedModel ======================

# Specify path to save the model
CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"

# Save model in TensorFlow SavedModel format
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

# ====================== Model conversion ======================
# ====================== no optimization or quantization ======================


import pathlib

# Load model into TFLiteConverter from previously saved directory
converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)

# Convert model to TFLite format
tflite_model = converter.convert()
# Specify directory to save the model
tflite_models_dir = pathlib.Path("/tmp/")  # temporary directory (can change path)

# Define path to model file
tflite_model_file = tflite_models_dir / 'model1.tflite'
# Save byte representation of model to file
tflite_model_file.write_bytes(tflite_model)


# ====================== Interpreter initialization ======================

from tqdm import tqdm  # Convenient progress bar in loops

# Load TFLite model and prepare interpreter
tflite_model_file = '/tmp/model1.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()  # Allocate memory for all tensors

# Get indices of input and output tensors
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []          # Model predictions will be stored here
test_labels, test_imgs = [], []  # Labels and images themselves (for analysis and visualization)

# Run 100 images through the model
for img, label in tqdm(test_batches.take(100)):  # Take 100 images from test dataset
    interpreter.set_tensor(input_index, img)     # Set input image
    interpreter.invoke()                         # Make prediction
    predictions.append(interpreter.get_tensor(output_index))  # Save result
    test_labels.append(label.numpy()[0])         # Save true label
    test_imgs.append(img)                        # Save the image itself

# Count number of correct predictions
score = 0
for item in range(0, len(predictions)):
    prediction = np.argmax(predictions[item])  # Index of max value = predicted class
    label = test_labels[item]                  # True label
    if prediction == label:
        score += 1
print("Out of 100 predictions I got " + str(score) + " correct")


# ====================== Utility functions for plotting ======================

#@title Utility functions for plotting
# Utilities for plotting

# Set class names (in same order as in dataset)
class_names = ['cat', 'dog']

# Function to display one image with prediction
def plot_image(i, predictions_array, true_label, img):
    # Get required element from arrays
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]

    plt.grid(False)       # Remove grid
    plt.xticks([])        # Remove X axis
    plt.yticks([])        # Remove Y axis

    img = np.squeeze(img) # Remove extra axes (if any), e.g. (1, 224, 224, 3) → (224, 224, 3)

    plt.imshow(img, cmap=plt.cm.binary)  # Show image

    predicted_label = np.argmax(predictions_array)  # Find predicted class (argmax of probabilities)

    # Label color: green — correct, red — incorrect
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    # Add label:
    # - predicted class
    # - probability (in percent)
    # - actual class
    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100*np.max(predictions_array),
            class_names[true_label]
        ),
        color=color
    )

# ====================== Visualization of some model results ======================

#@title Visualize the outputs { run: "auto" }
max_index = 73 #@param {type:"slider", min:1, max:100, step:1}
for index in range(0, max_index):
    plt.figure(figsize=(6, 3))         # Set size of each visualization
    plt.subplot(1, 2, 1)               # One subplot (second is not used)
    plot_image(index, predictions, test_labels, test_imgs)  # Show image with prediction
    plt.show()                         #