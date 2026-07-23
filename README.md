# 🧬 Colonoscopy Image Classification with Softmax Regression & K-Fold CV

A multiclass colonoscopy image classification project implemented **completely from scratch** using Python and NumPy, without relying on high-level machine learning frameworks (like Scikit-learn, PyTorch, or TensorFlow) for the core model logic.

The primary objective of this project is to implement and deeply understand the mathematical foundations behind multiclass classification, Softmax Regression, Gradient Descent, L2 Regularization, feature normalization without data leakage, K-Fold Cross-Validation for hyperparameter selection, and multi-class evaluation metrics.

---

## 🚀 Project Overview

This project utilizes extracted numerical features from colonoscopy images to classify samples into three distinct classes. 

The complete machine learning workflow includes:
- **Data Loading & Preprocessing:** Sample/patient-based data organization and feature handling.
- **Data Splitting:** Train / Development / Test dataset partitioning.
- **Normalization:** Train-only feature scaling to prevent data leakage.
- **Encoding:** One-Hot encoding for multiclass targets.
- **Algorithm Implementation:** Multiclass Softmax Regression built with NumPy.
- **Optimization:** Gradient Descent with L2 Regularization.
- **Validation:** K-Fold Cross-Validation for tuning Lambda.
- **Evaluation:** Loss tracking, Confusion Matrix, Precision, Recall, Macro F1-Score.

---

## 🧠 Model Architecture & Mathematical Foundations

### Softmax Regression
The classifier extends Logistic Regression to multiclass classification. For an input feature matrix **X**, the linear transformation is computed as:

> **Z = X · W + b**

Where:
* **X**: Input feature matrix
* **W**: Weight matrix
* **b**: Bias vector
* **Z**: Raw output scores (logits)

The **Softmax function** converts these raw scores into a probability distribution across classes:

> **P(class k) = exp(Z_k) / ∑ exp(Z_j)**

The predicted class corresponds to the class index with the highest computed probability:

> **ŷ = argmax(P)**

---

## 🛡️ L2 Regularization

To combat overfitting and constrain large weight values, **L2 Regularization (Ridge)** is integrated into the Cross-Entropy Loss function:

> **Cost = Cross_Entropy_Loss + (λ / (2 * m)) * ∑(W²)**

Where:
* **λ (Lambda)**: Regularization strength hyperparameter
* **m**: Number of training samples
* **W**: Model weight matrix

The regularization term is also included in the gradient computation during parameter updates:

> **dW = (1 / m) * Xᵀ · (P - Y) + (λ / m) * W**

---

## 🔍 Hyperparameter Selection (K-Fold Cross-Validation)

The optimal regularization strength (**λ**) was selected using **K-Fold Cross-Validation** on the development/training set. 

The evaluated candidate values were:  
* **λ ∈ {0.001, 0.01, 0.1, 1, 10, 100}**

### Validation Workflow:
1. The development dataset is split into **K** equal folds.
2. For each candidate **λ**, the model trains on **K - 1** folds and validates on the remaining fold.
3. Validation costs are averaged across all iterations.
4. The **λ** yielding the lowest mean validation loss is chosen for final model training.

---

## 📊 Feature Normalization & Preventing Data Leakage

Feature normalization is performed using statistics calculated **strictly from the training fold**:

> **X_normalized = (X - μ_train) / σ_train**

The computed mean (**μ_train**) and standard deviation (**σ_train**) are subsequently applied to scale the validation and unseen test datasets. This guarantees zero data leakage from evaluation sets into the training process.

---

## 📈 Model Training & Convergence

Parameters are updated iteratively using Gradient Descent. The loss curve was monitored throughout training to ensure numerical stability and convergence.

![Training Cost](https://github.com/Bilal-gul/Colonoscopy-Image-Classification-from-Scratch/blob/main/Colonoscopy_Prediction_ai/images/loss_curve.png)

---

## 🧪 Experimental Evaluation Results

After selecting the optimal **λ** via Cross-Validation, the final model was evaluated on a completely isolated test set.

### Overall Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **0.7000** |
| **Macro Precision** | **0.7778** |
| **Macro Recall** | **0.6667** |
| **Macro F1 Score** | **0.6556** |
| **Test Cost / Loss** | **1.9187** |

---

### 📌 Confusion Matrix

![Confusion Matrix](https://github.com/Bilal-gul/Colonoscopy-Image-Classification-from-Scratch/blob/main/Colonoscopy_Prediction_ai/images/confusion_matrix.png)

## 🗂️ Dataset Features & Domain Categories

The dataset consists of **698 numerical features** extracted from colonoscopy images, structured into three primary feature domains:

| Domain Category | Feature Count | Primary Characteristics Extracted |
| :--- | :---: | :--- |
| **2D Textural Features** | **422** | Surface patterns, local spatial variations, invariant textures |
| **2D Color Features** | **76** | Color distributions, hue variations, gray-level co-occurrences |
| **3D Shape Features** | **200** | Topological structures, geometric variations, surface dynamics |

---

## 🎨 Detailed Feature Sub-Groups

#### 1. 2D Textural Features (422 Total)
* **166** — AHT (*Autocorrelation Homogeneous Texture / Invariant Gabor Texture*)
* **256** — Rotational Invariant LBP (*Local Binary Patterns*)

#### 2. 2D Color Features (76 Total)
* **16** — Color Naming
* **13** — Discriminative Color
* **07** — Hue
* **07** — Opponent
* **33** — Color Gray-Level Co-occurrence Matrix (GLCM) Features

#### 3. 3D Shape Features (200 Total)
* **100** — ShapeDNA
* **100** — KPCA (*Kernel Principal Component Analysis*)

> **Note:** Combined, these feature groups provide a comprehensive multi-modal representation covering texture, color semantics, and three-dimensional geometrical properties of the tissue samples.

## 🏗️ Project Structure

```text
Colonoscopy_Prediction_ai/
│
├── data/
│   └── data.txt                    # Raw numerical features
│
├── images/
│   ├── training_cost.png           # Loss convergence plot
│   └── confusion_matrix.png        # Evaluated confusion matrix plot
│
├── model/
│    ├──metrics.py
│    ├──preprocessing.py
│    ├──L2_softmax_regression_model.py  # Core Softmax + L2 class implementation
│ 
├── choose_lambda.py                    # K-Fold CV logic for tuning lambda
├── train_test.py                       # Model training workflow and test dataset evaluation script
│
└── README.md                           # Project documentation
```

## 🛠️ Technologies & Tools Used

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized%20Math-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge&logo=python&logoColor=white)

- **Python 3.x:** Core programming language.
- **NumPy:** Vectorized matrix calculus, linear algebra, and manual gradient updates.
- **Pandas:** Data structuring, loading, and batch manipulation.
- **Matplotlib / Seaborn:** Plotting cross-entropy loss curves and confusion matrices.

---

## 🎯 Learning Objectives & Key Takeaways

Developed as a hands-on exploration of machine learning fundamentals, this project yielded critical engineering insights:

* 📐 **First-Principles Math:** Direct mathematical implementation of multiclass Softmax logit probabilities, loss calculations, and gradient updates.
* ⚖️ **Hyperparameter Tuning:** Practical experience in selecting L2 regularization parameters ($\lambda$) via K-Fold Cross-Validation.
* 🧼 **Strict Data Hygiene:** Preventing data leakage by deriving normalization statistics ($\mu, \sigma$) purely from the training folds.
* 🏥 **Medical Feature Semantics:** Understanding how domain-extracted features (Texture, Color, 3D Topology) map into multi-dimensional decision boundaries for medical image classification.

---

## 👨‍💻 Author

Developed as an in-depth machine learning project focused on building classification algorithms **from mathematical first principles**.

⭐ **If you found this implementation or README useful, feel free to give this repository a star!**
