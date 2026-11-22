# Machine Learning Lab — 11 Python Programs

Short, beginner-friendly examples of **11 machine learning methods** in Python. Every program reads an **external CSV file** with `pandas.read_csv()`, preprocesses data, fits or transforms a model, and plots a result using Matplotlib.

## Included programs

| File | Topic | CSV dataset | Plot |
| --- | --- | --- | --- |
| `01_linear_regression.py` | Linear Regression | `diabetes.csv` | Actual vs. predicted |
| `02_decision_tree.py` | Decision Tree | `iris.csv` | Decision tree |
| `03_knn.py` | K-Nearest Neighbors | `iris.csv` | Confusion matrix |
| `04_naive_bayes.py` | Naive Bayes | `iris.csv` | Confusion matrix |
| `05_ffnn.py` | Feed-Forward Neural Network | `iris.csv` | Training loss |
| `06_lda.py` | Linear Discriminant Analysis | `iris.csv` | 2D projection |
| `07_cnn.py` | Convolutional Neural Network | `digits.csv` | Accuracy by epoch |
| `08_cca.py` | Canonical Correlation Analysis | `iris.csv` | Correlation scatter |
| `09_fnn.py` | Fuzzy Neural Network (simplified) | `iris.csv` | Training loss |
| `10_gmm.py` | Gaussian Mixture Model | `iris.csv` | Clusters |
| `11_pca.py` | Principal Component Analysis | `iris.csv` | 2D projection |

## Run locally

1. Download or clone this repository.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run a program **from the repository folder**:

   ```bash
   python 01_linear_regression.py
   ```

For the CNN example, also install TensorFlow:

```bash
python -m pip install tensorflow
python 07_cnn.py
```

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the relevant `.py` and `.csv` files using the left-side **Files** panel, or use `from google.colab import files; files.upload()`.
3. To run a script in a code cell, use `%run 01_linear_regression.py` (replace the filename as needed).

Uploaded files are temporary and may disappear when Colab resets. Alternatively, clone your published GitHub repository in Colab:

```python
!git clone https://github.com/Amalmm12/ml.git
%cd ml
%run 01_linear_regression.py
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## CSV file formats

- `iris.csv`: 4 numerical flower measurements + `label` (classes 0, 1, 2).
- `diabetes.csv`: 10 numerical features + numeric `target`.
- `digits.csv`: 64 pixel columns + `label` (digits 0–9).

These CSVs were exported from standard scikit-learn learning datasets, then loaded as files by the programs (rather than imported through `load_iris`, `load_diabetes`, or `load_digits`). To substitute your own CSV, update the filename, target column, and feature-preparation steps.

**Note:** `09_fnn.py` uses Gaussian fuzzy membership features followed by an MLP. It is a simple fuzzy/neural hybrid, not a complete ANFIS implementation.

## Lab notes

The printable lab PDF is available separately from the project author.