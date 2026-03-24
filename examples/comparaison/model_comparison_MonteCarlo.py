import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from amelio_cp import Process
import time
import shap

#%% DATA LOADING & PREPROCESSING
data_path_input = input("Enter the path to the dataset's folder: ")
data_path = data_path_input + "/all_data_28pp_gps.csv"
feature_names_path = "amelio_cp/processing/Features13.xlsx"


print(time.time())
features_list, features_names = Process.prepare_features_list(feature_names_path)
print(f"Features used: \n", features_names)
X, y = Process.prepare_data2(data_path, "svc", '6MWT', features_list)


#%%MACHINE LEARNING MODELS INITIALISATION
# 3 classifiers have been compared
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}
colours = {
    'Random Forest': '#1f77b4',         # Blue
    'Support Vector Machine': '#ff7f0e', # Orange
    'Logistic Regression': '#2ca02c'    # Green
}
n_iterations = 100

print(f"Running {n_iterations} different Train/Test splits to assess variance...")

results = {} # Dictionary to store the results for comparison

#%% MODELS' TRAINING & EVALUATION
# Using area under the curve (AUC) to assess the models

plt.figure(figsize=(9, 5))

for name, model in models.items():
    print("\n" + "* " * 20)
    print(f"Running for {name}")
    
    # Build the Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ]) # Pipeline ensures SMOTE only applied to the training folds in EACH split

    test_auc_scores = []
    for i in range(n_iterations):
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=i)
        pipeline.fit(x_train, y_train)
            
        # Evaluating the model
        y_pred_proba = pipeline.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        test_auc_scores.append(auc)
      
    
    results[name] = test_auc_scores

    print(f"\n--- Model Stability ({name}) ---")
    print(f"Mean AUC:             {np.mean(test_auc_scores):.4f}")
    print(f"Stability (Std Dev):  {np.std(test_auc_scores):.4f}")
    print(f"Minimum AUC observed: {np.min(test_auc_scores):.4f}")
    print(f"Maximum AUC observed: {np.max(test_auc_scores):.4f}")    

    sns.histplot(
        test_auc_scores, 
        bins=15, 
        kde=True, 
        color=colours[name], 
        alpha=0.4, 
        label=f'{name} (Mean: {np.mean(test_auc_scores):.2f})'
    )
    plt.axvline(np.mean(test_auc_scores), color=colours[name], linestyle='dashed', linewidth=1.5, label=f'Mean AUC ({name}): {np.mean(test_auc_scores):.2f}')

    # fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
plt.title('Model accuracies across 100 Data Splits\n (speed)', fontsize=13, pad=15)
plt.xlabel('ROC AUC Score on hold-out test data', fontsize=12)
plt.ylabel('Frequency (dep. on nb of splits)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%% VISUALISATION OF THE MODELS' VARIABILITY
results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=results_df, 
    palette="pastel", 
    showfliers=False # Outliers hidden, so won't overlap with the stripplot points
)

sns.stripplot(
    data=results_df, 
    color='black', 
    alpha=0.6, 
    jitter=True, 
    size=6
)

plt.title(f'Assessment of Model Stability Across 25 Random Data Splits', fontsize=14, pad=15)
plt.ylabel('ROC AUC Score', fontsize=12)
plt.xlabel('Machine Learning Algorithm', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0.4, 1.05)

plt.tight_layout()
plt.show()


# #%% FEATURE IMPORTANCE ANALYSIS (if RF was good)

# # rf_model = models['Random Forest']
# # feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# # print("Top Predictors of Ambulatory Progress:")
# # print(feature_importances.sort_values(ascending=False).head(5))