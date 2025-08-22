# 🧠 MNIST Digit Classification — From‑Scratch NN & PyTorch CNN

## 1. Introduction
This repository explores handwritten digit recognition on the MNIST dataset using two complementary approaches:
1) a **neural network implemented from scratch** with NumPy/SciPy, and
2) a **convolutional neural network (CNN)** built with PyTorch.

The goal is to contrast fundamentals (manual forward/backprop & optimization) with a modern deep‑learning baseline.

---

## 2. Repository Structure
```
.
├── mnist_nn_fromscratch.ipynb   # From-scratch MLP (NumPy + SciPy)
├── cnn_mnist.py                 # PyTorch CNN
├── digitnn_results.json         # Saved test metrics (from-scratch run)
├── params.pickle                # Trained weights/metadata (from-scratch run)
├── data/
│   └── mnist_all.mat            # (Optional) MNIST .mat file if small & licensed
├── outputs/                     # (Optional) Plots / confusion matrices
├── requirements.txt
└── README.md
```
> If `mnist_all.mat` exceeds GitHub’s size limit or license constraints, omit it from the repo and follow the download instructions below.

---

## 3. Dataset
- **MNIST**: 70k grayscale 28×28 digit images (0–9).
- Typical split used in this repo: **50k train / 10k validation / 10k test**.
- If you don’t commit the dataset, create a local folder and place it there:
```
mkdir -p data
# Put mnist_all.mat in ./data/
```
- Alternatively, adjust `cnn_mnist.py` to use `torchvision.datasets.MNIST` which downloads data automatically.

---

## 4. Methods

### 4.1 From‑Scratch Neural Network (`mnist_nn_fromscratch.ipynb`)
- Single hidden layer MLP implemented with **NumPy**.
- Components: weight init, **sigmoid** activation, forward pass, **backpropagation**, L2 **regularization**.
- Trained with **Conjugate Gradient** (`scipy.optimize.minimize`).
- Includes basic feature selection (removing near-constant features).
- Saves artifacts to `digitnn_results.json` and `params.pickle`.

### 4.2 Convolutional Neural Network (`cnn_mnist.py`)
- **PyTorch** model: two 5×5 conv layers (ReLU + MaxPool) → FC(128) → logits(10).
- **Adam** optimizer, mini-batches.
- Script reports **test accuracy**; you can extend it to log training/validation metrics.

---

## 5. Results
> Fill in your measured numbers after running locally.

| Model                                | Test Accuracy | Training Time |
|--------------------------------------|---------------|---------------|
| From‑scratch NN (NumPy + SciPy)      | XX%           | YY sec        |
| CNN (PyTorch)                         | ZZ%           | WW sec        |

Notes:
- The CNN typically outperforms the vanilla MLP on image data.
- The from-scratch NN highlights understanding of learning mechanics without high-level frameworks.

---

## 6. Reproducibility

### 6.1 Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 6.2 From‑Scratch Notebook
```bash
jupyter notebook mnist_nn_fromscratch.ipynb
```
- Run all cells. Artifacts are saved to `digitnn_results.json` and `params.pickle`.

### 6.3 PyTorch CNN
```bash
python cnn_mnist.py
```
- By default it will train and print test accuracy at the end.
- To enable/adjust torchvision download path, edit the root path in the script if needed.

---

## 7. Files & Artifacts
- **`digitnn_results.json`**: JSON with model name, training time, and test accuracy.
- **`params.pickle`**: Pickled dictionary of trained weights / metadata.
- **`outputs/`**: Optional plots (loss curves, confusion matrices, misclassified samples).

---

## 8. Requirements
See `requirements.txt` for exact versions. Core libraries: **numpy, scipy, matplotlib, torch, torchvision, scikit‑learn, jupyter**.

---

## 9. License
If you choose to open-source, add a `LICENSE` (MIT is common).

---

## 10. Acknowledgments
- LeCun, Y. et al. **MNIST** database.
- PyTorch & SciPy documentation for reference implementations and APIs.
