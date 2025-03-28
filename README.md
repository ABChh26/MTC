# README

## Introduction

The main goal of this project is to analyze the relationship between medullary thyroid carcinoma (MTC) and associated gene/clinical data using machine learning. We employ a RandomForestClassifier to classify and evaluate models on various test datasets, and we use SHAP to interpret feature importance.

### Core Workflow

1. **Data Loading**: Load gene features and clinical data from multiple Excel files.
2. **Data Preprocessing**: Fill missing values, convert categorical variables (e.g., `tumor_grade`, `sex`, `Heredity`, `T`, `N`, `M`, `TNM stage`) into numerical codes.
3. **Feature Selection**: Use RandomForestClassifier together with `SelectFromModel` to identify the most discriminative features.
4. **Model Training & Evaluation**:
   - Run K-fold cross-validation to choose the best model on the training set.
   - Evaluate on independent test sets to output performance metrics (precision, recall, confusion matrix, AUC, etc.).
5. **SHAP Interpretability**: Compute SHAP values to evaluate the contribution and impact of each feature on model predictions.

---

## Project Structure

```
your_project/
│
├─ data/           # Folder containing the Excel data files
│   ├─ MTC_38features_20241230.xlsx
│   ├─ MTC_clinical_20240831.xlsx
│   ├─ MTC_gene_20240831_only_one.xlsx
│   ├─ MTC_all_info_20240912.xlsx
│   └─ ...
│
├─ main.py         # Python script containing the core logic
└─ README.md
```

> **Note**: The actual names of the Excel files depend on your local files.  
> The script assumes they reside in the `data/` folder.

---

## Requirements

- Python 3.7+
- Key Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `shap`
  - `argparse`
  
Install the dependencies via `pip`:

```bash
pip install pandas numpy scikit-learn shap argparse
```

---


---

## Usage

1. **Clone or Download**  
   Clone or download this project to your local environment. Ensure the `data/` directory contains the required Excel files.

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install pandas numpy scikit-learn shap argparse
   ```

3. **Run the Script**  
   From the terminal, navigate to the project folder and execute:
   ```bash
   python main.py --seed 42
   ```
   **Argument details:**
   - `--seed`: Random seed (default is 10).

3. **Output**  
   The script outputs:
   - Cross-validation scores and the best validation score.
   - Test set performance metrics (precision, recall, confusion matrix, AUC).
   - Prediction results (predicted labels, probabilities, true labels).
   - Selected important features.
   - SHAP values for feature interpretation.

---

## Key Functions

- **`load_data()`**  
  Loads data from multiple Excel files into dictionaries for further processing.

- **`print_metrics(y_true, y_pred, y_pred_proba)`**  
  Calculates and displays classification metrics including precision, recall, confusion matrix, and AUC.

- **`print_metrics_conf_matrix(y_test, y_pred, y_pred_proba)`**  
  Similar to `print_metrics` but emphasizes the confusion matrix.

- **`train(gene_discovery_data, gene_test1_data, gene_test2_data)`**  
  Trains a RandomForestClassifier with K-fold cross-validation, selects the best model, and evaluates on test datasets.

- **`Clinical_train_with_slectfeature(gene_discovery_data, gene_test1_data, gene_test2_data, max_features=10, random_state=1)`**  
  Performs feature selection, model training, evaluation, and SHAP-based feature importance analysis.

---

## Interpreting Results

- **Cross-validation**  
  - Outputs accuracy scores for each fold in K-fold CV.
  - Selects the model with the best overall cross-validation accuracy.

- **Test Set Performance**  
  Metrics include precision, recall, F1 Score, confusion matrix, and AUC.

- **Feature Importance**  
  Evaluates feature importance using both RandomForest’s built-in feature importances and SHAP values. Higher SHAP values indicate more influential features.

---

## Frequently Asked Questions (FAQ)

- **Excel File Names or Versions Differ**  
  Update file paths and sheet names in `load_data()` accordingly.

- **Library Compatibility Issues**  
  Use virtual environments (e.g., conda, venv) to resolve potential dependency conflicts.

- **Missing Data Issues**  
  Consider alternative strategies (e.g., median, mean, interpolation) if current filling strategy is unsuitable.

- **Categorical Variable Encoding**  
  Update the dictionary `r_dict` in the script if category mappings need to be modified.

---

## Contributing & Extensions

You are encouraged to modify the code by trying different models, feature selection strategies, or visualizations. Contributions, suggestions, and improvements are welcome via issues or pull requests.

---

## License

You can release this project under your laboratory’s or company’s preferred license. If not specified, consider using the [MIT License](https://opensource.org/licenses/MIT).

---

_We hope this README helps you and others easily understand and use the project. For further questions or clarifications, please reach out._

