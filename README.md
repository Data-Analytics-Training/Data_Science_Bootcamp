# Data Science Bootcamp (TechIstanbul)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](README.md)

A complete, project-based Python bootcamp covering the fundamentals of data science and machine learning — from basic programming to regression, clustering, and deep learning. This repository contains the course notebooks (*chapters*) and homework assignments (*oddev*) produced during the **TechIstanbul Data Science Bootcamp 2026**, taught by **Battal KOÇ**.

> The notebooks are written in Python and can be run from Google Colab, Jupyter Lab/Notebook, or VS Code. Each notebook is designed to be run sequentially from `chapter01` onward; the assignments follow the topics they cover.

## Contents

- [Course Content](#course-content)
- [Assignments](#assignments)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- [Environment & Requirements](#environment--requirements)
- [Project Structure](#project-structure)
- [How to Run the Notebooks](#how-to-run-the-notebooks)

## Course Content

| # | File | Topic | Key Tools & Concepts | Dataset |
|---|------|-------|---------------------|---------|
| 1 | `chapter01.ipynb` | Introduction to Python: Variables & Data Types | `print`, variables, data types (`int`, `float`, `str`) | — |
| 2 | `chapter02.ipynb` | Control Flow: Conditions (`if`) | Conditionals and logical checks (password validation example) | — |
| 3 | `chapter03.ipynb` | Error Handling: Try / Except / Finally | Exception handling and input validation (safe division) | — |
| 4 | `chapter04.ipynb` | Data Visualization | matplotlib, seaborn (scatter plots, distributions) | `Advertising.csv` |
| 5 | `chapter05.ipynb` | Multiple Linear Regression | `sklearn.linear_model.LinearRegression`, metrics, train/test split | `Advertising.csv` |
| 6 | `chapter06.ipynb` | Simple Linear Regression (Normal Equation & Gradient Descent) | numpy, manual gradient descent | Synthetic dataset |
| 7 | `chapter07.ipynb` | Classification & Clustering I | Logistic Regression, KNN, SVM (SVR), Neural Network, K-Means | — |
| 8 | `chapter08.ipynb` | Hierarchical Clustering, RNN & Text Classification (Phishing Emails) | Keras, sklearn.cluster, tokenization and embeddings | `imbd_dataset.csv` |
| 9 | `chapter09.ipynb` | Convolutional Neural Networks (CNNs) for Image Recognition | Keras, CNN architecture, confusion matrix | Fashion-MNIST |
| 10 | `chapter10.ipynb` | Template (placeholder for upcoming lessons) | — | — |

> The original Turkish filenames (`chapter`, `oddev`) are retained to stay consistent with the course materials; topic descriptions in this table and elsewhere are provided in English.

## Assignments

| # | File | Topic | Key Tools & Concepts | Dataset |
|---|------|-------|---------------------|---------|
| 1 | `oddev_1.ipynb` | Guess-the-Number Game | Basic input/output, loops (`while`) | — |
| 2 | `oddev_2.ipynb` | Library Management System | Functions, data structures (books) | — |
| 3 | `oddev_3.ipynb` | Exception Handling from a Text File | Try/except, file I/O (`kutuphane.txt`) | — |
| 4 | `oddev_4.ipynb` | Linear Regression: Study Hours vs. Exam Score | `LinearRegression`, MSE / MAE / RMSE / R² | `ogrenci_calisma_notlari.csv` |
| 5 | `oddev_5.ipynb` | Regression Models Comparison: Linear, Ridge, PCA, PLS | `sklearn.linear_model`, PCA, PLSRegression | Diabetes dataset |
| 6 | `oddev_6.ipynb` | Linear Regression vs. Artificial Neural Network (Regression) | `LinearRegression`, MLPRegressor | `enerji_tuketimi.csv` |
| 7 | `oddev_7.ipynb` | K-Means Clustering: Customer Segmentation | KMeans, silhouette score, elbow method | `musteri_segmentasyonu.csv` |
| 8 | `oddev_8.ipynb` | RNN Models Comparison: SimpleRNN, GRU, LSTM (Sentiment Analysis) | Keras, tokenization, embeddings | `imbd_dataset.csv` |
| 8b | `oddev_8b.ipynb` | Model Evaluation Visualizations (Training History, Confusion Matrix, ROC/AUC) | matplotlib, seaborn, `sklearn.metrics` | — |
| 9 | `oddev_9.ipynb` | CNN for Image Classification (Fashion-MNIST) | Keras, Fashion-MNIST, confusion matrix | Fashion-MNIST |

> `oddev_8` and `oddev_8b` are a matched pair: the former trains the recurrent models, while the latter visualizes and evaluates them.

## Datasets

| File | Description | Rows | Notes |
|-----|-------------|------|-------|
| `Advertising.csv` | TV, radio & newspaper ad budgets vs. sales | 200 | Course dataset; contains an unused index column. |
| `ogrenci_calisma_notlari.csv` | Study hours vs. exam score | 100 | Synthetic student dataset. |
| `musteri_segmentasyonu.csv` | Annual income & spending score | 250 | Turkish customer dataset. |
| `memekanseri.data` | Breast cancer diagnostic dataset (`age`, `tumor_size`, `tumor_perimeter`, target) | 306 (4 columns, no header) | Target: `1` = malignant, `2` = benign. Numeric values use comma decimals (European format). |
| `Phishing_Email.csv` | Email dataset for spam/phishing classification (with adversarial examples) | — | Very large; not committed to Git. |
| `imbd_dataset.csv` | IMDB 50k movie-review dataset (sentiment analysis) | — | Very large; not committed to Git. |

> Datasets are loaded from a local path when run in an IDE, and from Google Drive (`/content/drive/MyDrive/TechIstanbul/<filename>`) when run on Google Colab.

## Pretrained Models

| File(s) | Associated Task / Notebook | Notes |
|--------|---------------------------|-------|
| `rnn.keras`, `SimpleRNN_best.weights.h5` | RNN models (Sentiment Analysis) — `oddev_8.ipynb` | Regenerate via the notebook; large binary. |
| `model_GRU.keras`, `GRU_best.weights.h5` | RNN models (Sentiment Analysis) — `oddev_8.ipynb` | Regenerate via the notebook; large binary. |
| `model_LSTM.keras`, `LSTM_best.weights.h5` | RNN models (Sentiment Analysis) — `oddev_8.ipynb` | Regenerate via the notebook; large binary. |
| `cnn.keras`, `cnn_2.keras` | CNN models (Fashion-MNIST) — `oddev_9.ipynb` | Regenerate via the notebook; large binary. |

> The model binaries are large (up to 244 MB) and can be regenerated by running the corresponding notebooks, so they are excluded from Git. See [Environment & Requirements](#environment--requirements) for the needed packages and [How to Run the Notebooks](#how-to-run-the-notebooks) for reproduction instructions.

## Environment & Requirements

- Python 3.8+ (developed and tested on Python 3.12)
- Core libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `statsmodels`
- Deep learning: Keras / TensorFlow (required for the RNN and CNN assignments)

## Project Structure

```
Data_Science_Bootcamp/
│
├── (chapter*.ipynb)              # Course notebooks: Python basics → deep learning
├── (oddev*.ipynb)               # Homework assignments, grouped by topic
│
├── veriseti/                    # Local image datasets for the CNN assignment
│   ├── train/                  # Labeled training images (cat/dog) + metadata (.json)
│   └── test/                   # Unlabeled test images (cat/dog) + metadata (.json)
│
├── boyutlandirici.py            # Utility: GUI app to batch-resize images (tkinter + Pillow)
│
├── *.keras                     # Trained Keras model checkpoints (regenerable; excluded from Git)
├── *.weights.h5                # Saved model weights (regenerable; excluded from Git)
├── Advertising.csv             # Course dataset: ad budgets vs. sales
├── imbd_dataset.csv            # IMDB movie reviews (sentiment); excluded from Git
├── memekanseri.data            # Breast cancer diagnostic dataset (CSV-like)
├── musteri_segmentasyonu.csv   # Customer segmentation: income & spending score
├── ogrenci_calisma_notlari.csv # Student study hours vs. exam score
├── Phishing_Email.csv          # Email spam/phishing classification dataset
│
├── 6.png, 7.png, 8.png          # Supporting images (tutorial figures)
├── data.txt                   # Branding text file
│
├── LICENSE                     # MIT License (© 2026 Data Analytics Training Courses)
└── README.md                   # This file
```

## How to Run the Notebooks

1. Clone or download this repository into your working directory.
2. Place each dataset either in the repository root, or — when running on Google Colab — mount Google Drive and copy them under `/content/drive/MyDrive/TechIstanbul/`.
3. Open the notebooks in Jupyter Lab, VS Code, or Google Colab and run them **sequentially**, starting from `chapter01` (each builds on the previous one).
4. Run the assignments after the topics they cover, then the evaluation visualizations (`oddev_8b`, `oddev_9`).
5. For the deep-learning notebooks (`chapter08`, `chapter09` and related exercises), a GPU-enabled environment is recommended for faster training.

## License & Attribution

This project was prepared as part of the **TechIstanbul Data Science Bootcamp 2026**, taught by **Battal KOÇ**. The code and accompanying documentation are distributed under the [MIT License](LICENSE).
