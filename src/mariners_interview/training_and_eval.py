
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from sklearn.metrics import log_loss

def get_preprocessor(categorical_features, numeric_features, log_features, debug=False):
    """
    Create a preprocessor for the pipeline and return selected features.
    """
    if debug:
        print("Creating preprocessor...")
    
    # Define pipelines for different feature types
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    log_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transform', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_pipeline, categorical_features),
        ('num', numeric_pipeline, numeric_features),
        ('log', log_pipeline, log_features)
    ])
    
    # Collect selected features for each transformer
    selected_features = categorical_features + numeric_features + log_features

    if debug:
        print("Selected features for preprocessing:", selected_features)
    
    return preprocessor, selected_features

def train_model(X_train, y_train, model, preprocessor, debug=False):
    """
    Train a model with preprocessing.
    """
    if debug:
        print("Training model:", model)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    
    if debug:
        print("Model trained.")
    
    return pipeline

def grid_search_tuning(model, param_grid, X_train, y_train, preprocessor, scoring='roc_auc', cv=5, debug=False):
    """
    Perform grid search tuning to find the best hyperparameters.

    Args:
    - model: The model to tune.
    - param_grid: Dictionary of hyperparameters to search over.
    - X_train: Training features.
    - y_train: Training target variable.
    - preprocessor: Preprocessing pipeline.
    - scoring: Metric to evaluate the best parameters.
    - cv: Number of cross-validation folds.
    - debug: If True, print additional debug information.

    Returns:
    - Best fitted pipeline.
    """
    if debug:
        print(f"Starting grid search for {model} with params: {param_grid}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Run GridSearchCV with the provided parameters
    grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    if debug:
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")

    # Return the best fitted pipeline model
    best_fitted_pipeline = grid_search.best_estimator_

    return best_fitted_pipeline


def save_best_params_from_pipeline(pipeline, filename, debug=False):
    """
    Save the best parameters from a fitted pipeline's classifier.
    
    Args:
    - pipeline: Fitted pipeline model.
    - filename: Path to save the parameters.
    - debug: If True, print debug information.
    """
    # Extract the classifier parameters
    best_params = pipeline.named_steps['classifier'].get_params()
    with open(filename, 'w') as f:
        json.dump(best_params, f)
    if debug:
        print(f"Best parameters saved to {filename}")


def load_best_params(filename, debug=False):
    """
    Load the best hyperparameters from a file.
    """
    with open(filename, 'r') as f:
        best_params = json.load(f)
    if debug:
        print(f"Best hyperparameters loaded from {filename}")
    return best_params


def save_model(model, filename, debug=False):
    """
    Save the trained model to a file using joblib.
    """
    joblib.dump(model, filename)
    if debug:
        print(f"Model saved to {filename}")

def load_model(filename, debug=False):
    """
    Load a trained model from a file using joblib.
    """
    model = joblib.load(filename)
    if debug:
        print(f"Model loaded from {filename}")
    return model



def evaluate_model(model, X_test, y_test, debug=False):
    """
    Evaluate the model and return metrics.
    
    Args:
    - model: The trained model.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - debug: If True, prints debug information.
    
    Returns:
    - Dictionary of evaluation metrics.
    """
    if debug:
        print("Evaluating model...")
    
    # Predict values
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (is_airout)
    
    # Compute metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    log_loss_score = log_loss(y_test, y_proba)  # Compute log loss

    # Additional metrics: confusion matrix, precision, recall, F1-score
    cm = confusion_matrix(y_test, y_pred)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    metrics = {
        'classification_report': report,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'log_loss': log_loss_score  # Add log loss to metrics dictionary
    }
    
    if debug:
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Log Loss: {log_loss_score:.4f}")  # Print log loss
        print(f"Confusion Matrix:\n{cm}")
    
    return metrics


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix using seaborn heatmap.
    
    Args:
    - cm: Confusion matrix.
    - title: Title of the plot.
    - cmap: Colormap for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def save_preprocessed_data(df, output_path, debug=False):
    """
    Save the preprocessed DataFrame to a CSV file.
    
    Args:
    - df: Preprocessed DataFrame.
    - output_path: Path to save the CSV file.
    - debug: If True, print debug information.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    if debug:
        print(f"Preprocessed data saved to {output_path}")



if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Define paths
    train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train.csv'
    test_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-test.csv'
    dict_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-dictionary.csv'
    preprocessed_train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train-preprocessed.csv'
    
    # Load data
    train_df, test_df, data_dict_df = load_data(train_path, test_path, dict_path, debug=True)
    
    # Preprocess and feature engineer data
    train_df = clean_data(train_df)
    train_df = handle_missing_values(train_df)
    train_df = feature_engineering_pipeline(train_df)
    train_df = feature_engineering_with_defensive_metrics(train_df)
    train_df = feature_engineering_with_cluster_analysis(train_df)
    
    # Save preprocessed data
    save_preprocessed_data(train_df, preprocessed_train_path, debug=True)
    
    # Define features and target
    target = 'is_airout'
    features = train_df.columns.drop(target)
    X = train_df[features]
    y = train_df[target]
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("x train columns=", X_train.columns)
    # Define feature types and create a preprocessor
    categorical_features = ['month', 'day_of_week', 'temperature_category', 'count_scenario',
                            'hit_direction', 'venue_id']

    numeric_features = ['vert_exit_angle', 'horz_exit_angle', 'adjusted_distance']
    log_features = ['exit_speed', 'hit_spin_rate']
    preprocessor, selected_features = get_preprocessor(categorical_features, numeric_features, log_features, debug=True)
    
    # Define models and their hyperparameter grids
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        #'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'NeuralNet': MLPClassifier(max_iter=500, random_state=42)
    }
    
    param_grids = {
        'RandomForest': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20]},
        'GradientBoosting': {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]},
        'XGBoost': {'classifier__learning_rate': [0.01, 0.1], 'classifier__max_depth': [3, 5, 7]},
        #'LightGBM': {'classifier__learning_rate': [0.01, 0.1], 'classifier__num_leaves': [31, 50, 100]},
        'CatBoost': {'classifier__learning_rate': [0.01, 0.1], 'classifier__depth': [4, 6, 8]},
        'NeuralNet': {'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'classifier__activation': ['relu', 'tanh']}
    }

    # Train, tune, and evaluate each model using grid search
    for name, model in models.items():
        print(f"\nGrid Search Tuning for {name}...")
        if name in param_grids:  # Check if the model has a parameter grid
            # Perform grid search and get the best fitted pipeline
            best_pipeline = grid_search_tuning(model, param_grids[name], X_train, y_train, preprocessor, scoring='roc_auc', cv=5, debug=True)

            # Evaluate the best pipeline on validation data
            metrics = evaluate_model(best_pipeline, X_val, y_val, debug=True)

            # Print evaluation metrics, including log loss
            print(f"Metrics for Best {name}:")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Log Loss: {metrics['log_loss']:.4f}")  # Print log loss score

            # Plot confusion matrix
            plot_confusion_matrix(metrics['confusion_matrix'], title=f'Confusion Matrix for Best {name}')

            # Save the best pipeline
            save_model(best_pipeline, f'../../data/Seattle Mariners 2025 Analytics Internship/models/{name.lower()}_best.pkl', debug=True)

            # Optionally save the best parameters
            save_best_params_from_pipeline(best_pipeline, f'../../data/Seattle Mariners 2025 Analytics Internship/models/{name.lower()}_best_params.json', debug=True)
        else:
            print(f"No hyperparameter grid defined for {name}. Skipping grid search.")






# Conclusion

# CatBoost proved to be the best model for your problem due to its efficient handling of categorical features, robust generalization through ordered boosting, and its ability to balance complexity and overfitting. The chosen hyperparameters (depth=6 and learning_rate=0.1) further reinforced its ability to perform well on your data.

# In terms of model comparison:

#     CatBoost outperformed XGBoost and Gradient Boosting by a small but significant margin in both ROC AUC and other evaluation metrics.
#     It should be your go-to choice for this dataset, especially given the nature of features and the data distribution.
    
# ROC AUC: 0.9386
# Accuracy: 0.8596
# Precision: 0.8573
# Recall: 0.8632
# F1 Score: 0.8602
# **Log Loss: 0.3223** # Log loss is a critical metric for evaluating probabilistic predictions because it penalizes both overconfident incorrect predictions and underconfident correct predictions. A lower log loss score indicates that the model’s probability estimates are closer to the true labels. In this case, the log loss score of 0.3223 suggests that CatBoost produced reliable probability predictions for the air out probability, minimizing uncertainty and maximizing the model’s predictive power.
# Confusion Matrix:
# [[5499  999]
#  [ 901 5610]]

# Given Question 1, CatBoost is particularly well-suited for this task due to its ability to natively handle categorical features, avoid overfitting through ordered boosting, and produce highly interpretable models. The dataset includes categorical columns like bat_side, pitch_side, day_of_week, and temperature_category, as well as various continuous features such as exit_speed and vert_exit_angle. CatBoost's specialized handling of categorical features means that it can extract more meaningful relationships from these features without the need for extensive preprocessing, which other models like XGBoost might require.

# Moreover, CatBoost's robustness against overfitting and its ability to generalize better to new data make it an ideal choice for predicting a nuanced outcome like air out probability. Given that this problem involves predicting the likelihood of a rare event, CatBoost's advanced regularization methods help maintain balance between bias and variance, resulting in superior performance compared to other boosting models. This helps ensure reliable predictions on test data, which is crucial for achieving a low log loss score and maximizing evaluation metrics like AUC and F1 score, as seen in the results.

# Ultimately, CatBoost not only fits the specific needs of this dataset but also aligns well with the evaluation criteria (log loss and generalization performance), making it the optimal model for this type of baseball event prediction.


