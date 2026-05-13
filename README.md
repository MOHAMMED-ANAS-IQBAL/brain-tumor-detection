# Brain Tumor Detection using Deep Learning

A deep learning system that classifies brain MRI scans into four categories and localises the tumour region using **Grad-CAM** visualisation. Includes both a training notebook and an interactive **Streamlit** web app.

---

## Classes

| Label | Description |
|---|---|
| `glioma` | Tumour originating in the glial cells |
| `meningioma` | Tumour arising from the meninges |
| `notumor` | No tumour detected |
| `pituitary` | Tumour in the pituitary gland |

---

## Dataset Structure

```
Brain Tumor Detection/
├── Training/
│   ├── glioma/          # 1400 images
│   ├── meningioma/      # 1400 images
│   ├── notumor/         # 1400 images
│   └── pituitary/       # 1400 images
└── Testing/
    ├── glioma/          # 400 images
    ├── meningioma/      # 400 images
    ├── notumor/         # 400 images
    └── pituitary/       # 400 images
```

Dataset source: [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Data Split

| Split | Source | Images | Per class |
|---|---|---|---|
| **Train** | `Training/` (80%) | 4 800 | 1 200 |
| **Validation** | `Training/` (20%) | 800 | 200 |
| **Test** | `Testing/` (separate) | 1 600 | 400 |

The model **never sees** `Testing/` images during training or validation — they are safe to use for real-world evaluation.

---

## Models

### TensorFlow / Keras (main notebook)
- File: `brain_tumor_detection_using_deep_learning.ipynb`
- Base model: VGG16 (ImageNet weights, last 3 conv layers fine-tuned)
- Input size: 128 × 128
- Saves: `best_model.keras`

### PyTorch (alternative notebook)
- File: `brain-tumor-detection-using-vgg16-pytorch.ipynb`
- Base model: VGG16 (torchvision)
- Input size: 224 × 224
- Saves: `best_model.pth`

---

## Features

- **Transfer Learning** — VGG16 pretrained on ImageNet
- **Grad-CAM** — highlights the tumour region the model focused on
- **Early Stopping** — stops training when validation loss plateaus
- **Learning Rate Scheduler** — halves LR after 3 stagnant epochs
- **Model Checkpoint** — saves only the best model (by val loss)
- **Consistent Preprocessing** — augmentation on training only; val/test/inference use plain normalisation
- **Colab + Local** — paths auto-detected; works on both platforms

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install tensorflow==2.15.0 scikit-learn matplotlib seaborn pillow numpy opencv-python streamlit
# For PyTorch notebook also run:
pip install torch torchvision
```

### 4. Add the dataset
Download from Kaggle and place the folders so the structure matches the one shown above.

---

## Usage

### Train the model (notebook)
1. Open `brain_tumor_detection_using_deep_learning.ipynb` in Jupyter or VS Code
2. Select **Python 3.11** kernel
3. Run all cells top to bottom
4. Trained model is saved as `best_model.keras`

### Run the Streamlit app
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.
Upload any MRI image from the `Testing/` folder (or your own scan) to get a prediction + Grad-CAM overlay.

---

## Results (TensorFlow model, 5 epochs on Colab)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| glioma | 0.97 | 0.98 | 0.98 |
| meningioma | 0.93 | 0.90 | 0.91 |
| notumor | 0.95 | 1.00 | 0.97 |
| pituitary | 0.93 | 0.91 | 0.92 |
| **Overall accuracy** | | | **0.95** |

---

## Project Structure

```
Brain Tumor Detection/
├── brain_tumor_detection_using_deep_learning.ipynb  # TensorFlow training notebook
├── brain-tumor-detection-using-vgg16-pytorch.ipynb  # PyTorch training notebook
├── app.py                                            # Streamlit web app
├── Training/                                         # Training images (not tracked by git)
├── Testing/                                          # Test images (not tracked by git)
├── .gitignore
└── README.md
```

> **Note:** Model weight files (`best_model.keras`, `best_model.pth`) and dataset folders are excluded from git because they are too large. Train the model locally after cloning.

---

## Disclaimer

This project is for research and educational purposes only. It is **not** a clinical diagnostic tool and should not be used as a substitute for professional medical advice.
