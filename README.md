# Parking-detection

# Context-Based Parking Slot Detection & Context Recognizer

This project provides code and tools for training and evaluating a YOLOv3-based parking slot detector and a context recognizer using TensorFlow 2.x. It is designed for both local (Windows) and Google Colab environments.

---

## Project Structure

- `context-based-parking-slot-detect/`  
  Main codebase for parking slot detection and context recognition.
- `deep-parking/`  
  PyTorch-based model (not main focus).
- `PIL-park/`  
  Contains raw images and labels for training/testing.
- `pre_weight/`  
  Pretrained weights (not tracked by git).
- `weight_pcr/`, `weight_psd/`  
  Model output weights (not tracked by git).

---

## Requirements

- Python 3.8+ (recommended)
- TensorFlow 2.x
- Keras
- tf_slim
- Other dependencies as needed (see below)

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install Dependencies
Create a virtual environment (recommended):
```sh
python -m venv .venv
.venv\Scripts\activate  # On Windows
# Or
source .venv/bin/activate  # On Linux/Mac
```
Install required packages:
```sh
pip install --upgrade pip
pip install tensorflow keras tf_slim
# Add any other requirements as needed
```

---

## Data Preparation

### 1. Prepare Raw Data
- Place your raw images and label files in the `PIL-park/train/image/`, `PIL-park/train/label/`, `PIL-park/test/image/`, and `PIL-park/test/label/` folders.

### 2. Generate TFRecords
Run the data preparation script to generate TFRecords from raw data:
```sh
python context-based-parking-slot-detect/prepare_tfrecords.py
```
- This will create TFRecord files in the appropriate folders.

---

## Training

### 1. Parking Slot Detector
Train the YOLOv3-based detector:
```sh
python context-based-parking-slot-detect/train.py --batch_size 16 --epochs 10
```
- Adjust `--batch_size` and `--epochs` as needed for your hardware.

### 2. Context Recognizer
Train the context recognizer:
```sh
python context-based-parking-slot-detect/parking_context_recognizer/train.py --batch_size 16 --epochs 10
```

---

## Inference & Evaluation

- After training, model weights will be saved in `weight_pcr/` and `weight_psd/`.
- Use the provided test scripts to evaluate or visualize results:
```sh
python context-based-parking-slot-detect/test.py
```

---

## Notes

- **Model weights, TFRecords, logs, and virtual environments are NOT included in the repository.**
  - You must generate TFRecords from raw data as described above.
  - You must train the models to obtain weights, or download them if available.
- If you want to use pretrained weights, place them in the `pre_weight/` folder.
- For Google Colab, upload your code and data, then follow the same steps above.

---

## Troubleshooting

- If you encounter missing module errors, ensure all dependencies are installed.
- If you run out of memory, reduce the batch size.
- For GPU support, ensure you have the correct TensorFlow version and drivers.

---

## Contact
For questions or issues, please open an issue on GitHub or contact the project maintainer.
