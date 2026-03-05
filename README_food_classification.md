# Ultra-Processed Food Classification

Binary classification model to predict whether a food product is **ultra-processed** based on its nutritional attributes, store, food category, and brand.

## Dataset

Product-level nutritional data with the FPro classification system (classes 0–3), where class 3 corresponds to ultra-processed foods. The binary target maps class 3 → 1 (ultra-processed) and classes 0–2 → 0 (not ultra-processed).

## Project Overview

The analysis follows a complete ML pipeline:

1. **Exploratory Data Analysis** — data structure, missing values, target distribution, descriptive statistics, and feature distributions with histograms and boxplots
2. **Preprocessing** — drop irrelevant columns, reduce brand cardinality (top 30 + "Other"), one-hot encoding for categoricals, standard scaling for numericals, balanced class weights for imbalance
3. **Model Training & Tuning** — four models compared with a 60/20/20 train/validation/test split:
   - Baseline (most frequent class)
   - Decision Tree (tuned: max_depth, min_samples_leaf)
   - Random Forest (tuned: n_estimators, max_depth, min_samples_leaf)
   - Logistic Regression (tuned: regularization C)
4. **Outlier Detection** — K-Means clustering on training features to identify and remove the top 5% noisiest samples, with before/after performance comparison
5. **Evaluation** — accuracy, precision, recall, F1, and AUC across all splits

## Methods & Tools

| Category | Details |
|---|---|
| **Language** | Python |
| **EDA** | Pandas, Matplotlib, Seaborn |
| **ML** | Scikit-learn (Decision Tree, Random Forest, Logistic Regression, K-Means, DummyClassifier) |
| **Imbalance** | Balanced class weights, RandomOverSampler (imblearn) |
| **Pipeline** | Scikit-learn Pipeline + ColumnTransformer |

## Key Findings

- **Random Forest** achieved the best overall F1 and AUC, capturing non-linear interactions between nutritional features.
- **Logistic Regression** performed competitively, suggesting a strong linear component in the decision boundary.
- **Class imbalance handling** via balanced weights was critical — without it, models defaulted to the majority class.
- **Outlier removal** had a modest effect, indicating the dataset is relatively clean.
- **Brand** and **food category** were important categorical predictors alongside nutritional attributes like sugar, sodium, and fat.

## Project Structure

```
├── Project.ipynb        # Complete analysis notebook
├── README.md            # Project documentation
└── .gitignore           # Excludes data files
```

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
   ```
3. Place `product_data.csv` in the project root
4. Run the notebook

## Author

Sebastian — M.S. Data Science & AI, Florida International University
