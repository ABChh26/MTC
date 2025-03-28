import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
import shap

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np  # 确保引入 numpy


import pandas as pd

def load_data():
    """
    Load data from multiple Excel files. Each Excel file contains one or more sheets.
    This function reads all sheets from each Excel file into dictionaries, 
    where the keys are the sheet names and the values are the corresponding DataFrames.

    Returns:
        gene_data (dict): Dictionary containing sheets from the gene features Excel file.
        clinical_data (dict): Dictionary containing sheets from the clinical data Excel file.
        gene_onlyone_data (dict): Dictionary containing sheets from the 'gene only one' Excel file.
        gene_all_data (dict): Dictionary containing sheets from the full gene info Excel file.
    """
    # File paths
    # Optionally used files:
    # file_gene = 'data/MTC_38features_20241206.xlsx'
    # file_gene = 'data/MTC_38features_20240925++++.xlsx'
    # file_gene = 'data/MTC_38features_20241011.xlsx'
    
    file_gene = 'data/MTC_38features_20241230.xlsx'
    file_clinical = 'data/MTC_clinical_20240831.xlsx'
    file_gene_onlyone = 'data/MTC_gene_20240831_only_one.xlsx'
    file_gene_all = 'data/MTC_all_info_20240912.xlsx'
    # file_gene_all = 'data/MTC_clini_info_20240912.xlsx'

    # Read all sheets from each Excel file into a dictionary.
    gene_data = pd.read_excel(file_gene, sheet_name=None)
    clinical_data = pd.read_excel(file_clinical, sheet_name=None)
    gene_onlyone_data = pd.read_excel(file_gene_onlyone, sheet_name=None)
    gene_all_data = pd.read_excel(file_gene_all, sheet_name=None)

    # Print the sheet names for debugging or verification.
    print("Gene Data Sheets:", gene_data.keys())
    print("Gene Only One Data Sheets:", gene_onlyone_data.keys())
    print("Clinical Data Sheets:", clinical_data.keys())
    print("Gene All Data Sheets:", gene_all_data.keys())

    return gene_data, clinical_data, gene_onlyone_data, gene_all_data



from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             confusion_matrix, 
                             roc_auc_score)

def print_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate and print various classification metrics.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_pred_proba (array-like): Predicted probabilities for positive class.
    
    Returns:
        float: The calculated AUC (Area Under the ROC Curve).
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    # Print selected metrics
    # print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:{conf_matrix}")
    # print(f"AUC: {auc}")

    return auc



def print_metrics_conf_matrix(y_test1, y_pred_test1, y_pred_proba_test1):
    accuracy = accuracy_score(y_test1, y_pred_test1)
    precision = precision_score(y_test1, y_pred_test1, average='weighted')
    recall = recall_score(y_test1, y_pred_test1, average='weighted')
    f1 = f1_score(y_test1, y_pred_test1, average='weighted')
    conf_matrix = confusion_matrix(y_test1, y_pred_test1)
    auc = roc_auc_score(y_test1, y_pred_proba_test1) 

    # print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    # print(f"AUC: {auc}")
    return conf_matrix[1,1]


def train(gene_discovery_data, gene_test1_data, gene_test2_data):
    """
    Train and validate a RandomForestClassifier using K-Fold cross-validation on the gene discovery dataset.
    Then evaluate the trained best model on two test datasets.

    Args:
        gene_discovery_data (pd.DataFrame): Training data containing 'MS_ID', 'label', and feature columns.
        gene_test1_data (pd.DataFrame): First test dataset, similarly containing 'MS_ID', 'label', and features.
        gene_test2_data (pd.DataFrame): Second test dataset, with the same columns.
    
    Returns:
        None
    """
    # Prepare training and test data
    X_train = gene_discovery_data.drop(columns=['MS_ID', 'label'])
    y_train = gene_discovery_data['label']

    X_test1 = gene_test1_data.drop(columns=['MS_ID', 'label'])
    y_test1 = gene_test1_data['label']

    X_test2 = gene_test2_data.drop(columns=['MS_ID', 'label'])
    y_test2 = gene_test2_data['label']

    # Initialize K-Fold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_score = 0
    best_model = None

    # Perform K-Fold cross-validation
    for train_index, val_index in kf.split(X_train):
        X_train_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_fold, y_train_fold)

        # Validate the model
        score = model.score(X_val_fold, y_val_fold)
        print(f"Validation Score: {score}")

        # Keep track of the best-performing model
        if score > best_score:
            best_score = score
            best_model = model

    print(f"Best Cross-validation Score: {best_score}")

    # Evaluate on the first test dataset
    print("Metrics on Test1 Data:")
    y_pred_proba_test1 = best_model.predict_proba(X_test1)[:, 1]  # Probability estimates for the positive class
    print_metrics(y_test1, best_model.predict(X_test1), y_pred_proba_test1)

    # Evaluate on the second test dataset
    print("Metrics on Test2 Data:")
    y_pred_proba_test2 = best_model.predict_proba(X_test2)[:, 1]  # Probability estimates for the positive class
    print_metrics(y_test2, best_model.predict(X_test2), y_pred_proba_test2)



def Clinical_train_with_slectfeature(gene_discovery_data, gene_test1_data, gene_test2_data, max_features=10, random_state=1):
    X_train = gene_discovery_data.drop(columns=['MS_ID', 'label'])
    y_train = gene_discovery_data['label']

    X_test1 = gene_test1_data.drop(columns=['MS_ID', 'label'])
    y_test1 = gene_test1_data['label']

    X_test2 = gene_test2_data.drop(columns=['MS_ID', 'label'])
    y_test2 = gene_test2_data['label']

    # concat , normalize the data, deconcate the data
    gene_data_all = pd.concat([X_train, X_test1, X_test2])
    
    # normalize the data with z-score
    gene_data_all = (gene_data_all - gene_data_all.mean()) / gene_data_all.std()
    
    X_train = gene_data_all.iloc[:gene_discovery_data.shape[0], :]
    X_test1 = gene_data_all.iloc[gene_discovery_data.shape[0]:gene_discovery_data.shape[0] + gene_test1_data.shape[0], :]
    X_test2 = gene_data_all.iloc[gene_discovery_data.shape[0] + gene_test1_data.shape[0]:, :]

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    best_score = 0
    best_model = None
    selected_features = None

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        feature_selector = RandomForestClassifier(random_state=random_state)
        feature_selector.fit(X_train_fold, y_train_fold)

        importances = feature_selector.feature_importances_
        feature_names = X_train_fold.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("Feature Importances:")
        print(importance_df)

        selector = SelectFromModel(feature_selector, threshold='median', max_features=max_features, prefit=False)

        X_train_fold_selected = selector.fit_transform(X_train_fold, y_train_fold)
        X_val_fold_selected = selector.transform(X_val_fold)

        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train_fold_selected, y_train_fold)

        score = model.score(X_val_fold_selected, y_val_fold)

        if score > best_score:
            best_score = score
            best_model = model
            selected_features = selector.get_support()
            X_train_best = X_train_fold.iloc[:, selected_features]
            y_train_best = y_train_fold

    print(f"Best Cross-validation Score: {best_score}")

    print(f"Best Cross-validation Score: {best_score}")

    feature_names = X_train.columns
    selected_feature_names = feature_names[selected_features]
    print(f"Selected Features: {selected_feature_names.tolist()}")

    X_test1_selected = X_test1.iloc[:, selected_features].to_numpy()
    X_test2_selected = X_test2.iloc[:, selected_features].to_numpy()

    print('-' * 80)
    print("Metrics on Test1 Data:")
    y_pred_test1 = best_model.predict(X_test1_selected)
    y_pred_proba_test1 = best_model.predict_proba(X_test1_selected)[:, 1] 
    print_metrics(y_test1, y_pred_test1, y_pred_proba_test1)

# Print prediction results for Test1 dataset
    print("Test1 dataset: comparison of predictions, prediction probabilities, and true labels:")
    for idx in range(len(y_test1)):
        ms_id = gene_test1_data['MS_ID'].iloc[idx]
        true_label = y_test1.iloc[idx]
        pred_label = y_pred_test1[idx]
        pred_proba = y_pred_proba_test1[idx]
        print(f"Sample ID: {ms_id}, True Label: {true_label}, Predicted Label: {pred_label}, Prediction Probability: {pred_proba:.4f}")

    print('-' * 80)
    print("Metrics on Test2 Data:")
    y_pred_test2 = best_model.predict(X_test2_selected)
    y_pred_proba_test2 = best_model.predict_proba(X_test2_selected)[:, 1]  # Probability for the positive class
    print_metrics(y_test2, y_pred_test2, y_pred_proba_test2)

    # Print prediction results for Test2 dataset
    print("Test2 dataset: comparison of predictions, prediction probabilities, and true labels:")
    for idx in range(len(y_test2)):
        ms_id = gene_test2_data['MS_ID'].iloc[idx]
        true_label = y_test2.iloc[idx]
        pred_label = y_pred_test2[idx]
        pred_proba = y_pred_proba_test2[idx]
        print(f"Sample ID: {ms_id}, True Label: {true_label}, Predicted Label: {pred_label}, Prediction Probability: {pred_proba:.4f}")

    print('-' * 80)
    print("Calculating SHAP values for feature importance ranking...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_best)

    shap_values = np.array(shap_values) 
    shap_values_mean = np.mean(np.abs(shap_values), axis=0) 
    shap_importance = np.mean(shap_values_mean, axis=1) 

    # import pdb; pdb.set_trace()
    shap_importance_df = pd.DataFrame({'Feature': selected_feature_names, 'SHAP Importance': shap_importance})
    shap_importance_df = shap_importance_df.sort_values(by='SHAP Importance', ascending=False)

    print("Feature Importance Ranking based on SHAP Values:")
    print(shap_importance_df)
    
    return shap_importance_df, X_train, X_test1, X_test2, y_train, y_test1, y_test2, gene_test1_data, gene_test2_data





if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    
    args = parser.parse_args()
    gene_data, clinical_data, gene_onlyone_data, gene_data_all = load_data()
    gene_data_all = gene_data


    print("===========================================")
    label_encoders = {}
    categorical_columns = ['tumor_grade', 'sex', 'Heredity', 'T', 'N', 'M', 'TNM stage']

    print("Training Model... on Clinical Data")
    clinical_discovery_data, clinical_test1_data, clinical_test2_data = clinical_data['Discovery'], clinical_data['test1'], clinical_data['test2']
    gene_data_all_discovery_data, gene_data_all_test1_data, gene_data_all_test2_data = gene_data_all['Discovery'], gene_data_all['test1'], gene_data_all['test2']
    
    r_dict = {
        'sex': {'F': 0, 'M': 1},
        'tumor_grade': {'low': 0, 'high': 1, None: -1},
        'Heredity': {'sporadic': 0, 'hereditary': 1},
        'T': {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4},
        'N': {'N0': 0, 'N1a': 1, 'N1b': 2},
        'M': {'M0': 0, 'M1': 1},
        'TNM stage': {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    }
    clinical_discovery_data = clinical_discovery_data.replace(r_dict)
    clinical_test1_data = clinical_test1_data.replace(r_dict)
    clinical_test2_data = clinical_test2_data.replace(r_dict)
    
    gene_data_all_discovery_data = gene_data_all_discovery_data.replace(r_dict)
    gene_data_all_test1_data = gene_data_all_test1_data.replace(r_dict)
    gene_data_all_test2_data = gene_data_all_test2_data.replace(r_dict)

    gene_data_all_discovery = gene_data_all_discovery_data
    gene_data_all_test1_data = gene_data_all_test1_data
    gene_data_all_test2_data = gene_data_all_test2_data


    gene_data_all_discovery_clean = gene_data_all_discovery.fillna(gene_data_all_discovery.min())
    gene_test1_data_all_clean = gene_data_all_test1_data.fillna(gene_data_all_test1_data.min())
    gene_data_all_test2_data_clean = gene_data_all_test2_data.fillna(gene_data_all_test2_data.min())
    
    shap_importance_df, X_train, X_test1, X_test2, y_train, y_test1, y_test2, gene_test1_data, gene_test2_data = Clinical_train_with_slectfeature(
        gene_data_all_discovery_clean, 
        gene_test1_data_all_clean, 
        gene_data_all_test2_data_clean, 
        max_features=20, 
        random_state=42
        )
    
    
    print(shap_importance_df['Feature'].to_list())
    feature_used = shap_importance_df['Feature'].to_list()

    
    X_train = X_train[feature_used]
    X_test1 = X_test1[feature_used]
    X_test2 = X_test2[feature_used]
    
    accuracy_score_list = []

    model = RandomForestClassifier(random_state=124)
    model.fit(X_train, y_train)
    
    y_pred_test1 = model.predict(X_test1)
    y_pred_proba_test1 = model.predict_proba(X_test1)[:, 1]  # 获取预测的概率分数（针对二分类情况）
    max = print_metrics_conf_matrix(y_test1, y_pred_test1, y_pred_proba_test1)
    print('auc max', max)
    
    accuracy_score_list.append(max)
    print('------>', np.max(accuracy_score_list), np.argmax(accuracy_score_list))

