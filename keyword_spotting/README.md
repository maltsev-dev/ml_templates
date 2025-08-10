# 🗣️ Keyword Spotting with TensorFlow

This project implements a complete keyword spotting (KWS) pipeline using the [Mini Speech Commands](https://www.tensorflow.org/datasets/catalog/mini_speech_commands) dataset.

---

## 📦 Project Structure

```

keyword\_spotting/
├── data/                  # Dataset will be downloaded here
├── models/                # Trained models and .tflite output
├── logs/                  # TensorBoard or training logs (optional)
├── scripts/
│   ├── download\_data.py   # Downloads the dataset
│   ├── preprocess.py      # Audio preprocessing pipeline
│   ├── train.py           # Model definition and training
│   ├── evaluate.py        # Evaluation on validation set
│   └── convert\_tflite.py  # Converts trained model to TFLite
├── utils/
│   └── audio\_utils.py     # Optional: waveform/spectrogram plotting
├── main.py                # CLI to run individual steps
├── run\_all.py             # Runs the full pipeline in order
└── requirements.txt       # Project dependencies

````

---

## ✅ Setup Instructions

### 1. Clone the project

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Full Pipeline

### Option 1: One-step execution

```bash
python run_all.py
```

This will:

* Download the dataset
* Train the model
* Evaluate validation accuracy
* Convert the model to `.tflite`

### Option 2: Step-by-step via CLI

```bash
python main.py download    # Download dataset
python main.py train       # Train the model
python main.py evaluate    # Evaluate accuracy
python main.py convert     # Convert to TFLite
```

---

## 📂 Output Files

| File                         | Description                      |
| ---------------------------- | -------------------------------- |
| `models/model.h5`            | Trained Keras model              |
| `models/model.tflite`        | Quantized TensorFlow Lite model  |
| `data/mini_speech_commands/` | Folder with downloaded WAV files |

---

## 🛠️ Requirements

* Python 3.8–3.11
* TensorFlow 2.x
* NumPy, Matplotlib, Seaborn, Scikit-learn, IPython

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT License. For educational use with TinyMLx and TensorFlow.