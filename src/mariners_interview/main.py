
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# from prediction import predict, save_predictions
# from training_and_eval import (
#     get_preprocessor, train_model, save_model, load_model, grid_search_tuning, evaluate_model, save_best_params_from_pipeline
# )
# from data_loading import load_data
# from preprocessing import clean_data, handle_missing_values
# from feature_engineering import feature_engineering_pipeline


def main():
    # Paths to data files
    train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train.csv'
    test_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-test.csv'
    dict_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-dictionary.csv'
    
    # Load data
    train_df, test_df, data_dict_df = load_data(train_path, test_path, dict_path, debug=True)
    
    # Preprocessing
    train_df = clean_data(train_df, debug=True)
    train_df = handle_missing_values(train_df, debug=True)
    
    # Define feature types for preprocessing
    categorical_features = ['month', 'day_of_week', 'temperature_category', 'count_scenario', 'hit_direction']
    numeric_features = ['vert_exit_angle', 'horz_exit_angle', 'adjusted_distance']
    log_features = ['exit_speed', 'hit_spin_rate']
    
    # Get preprocessor and selected features
    preprocessor, selected_features = get_preprocessor(categorical_features, numeric_features, log_features, debug=True)
    
    # Feature Engineering with filtering for train_df, include target variable
    train_df = feature_engineering_pipeline(train_df, selected_features=selected_features, include_target=True, debug=True)
    
    # Define target and features
    target = 'is_airout'
    features = selected_features
    X = train_df[features]
    y = train_df[target]
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    best_model_pipeline = CatBoostClassifier(depth=6, learning_rate=0.1, random_state=42, verbose=0)
    model_pipeline = train_model(X_train, y_train, best_model_pipeline, preprocessor, debug=True)
    
    metrics = evaluate_model(model_pipeline, X_val, y_val, debug=True)
    print(f"Metrics:")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Save model
    save_model(model_pipeline, f'../../data/Seattle Mariners 2025 Analytics Internship/models/catboost.pkl', debug=True)
    
    # Process test data
    print("test_df columns = ", test_df.columns)
    test_df = clean_data(test_df, debug=True)
    test_df_original = test_df.copy()
    test_df = feature_engineering_pipeline(test_df, selected_features=selected_features, include_target=False, debug=True)
    X_test = test_df[features]
    
    # Make predictions
    test_df['p_airout'] = predict(model_pipeline, X_test, debug=True)
    
    # Combine predictions with original test data
    test_df_original = test_df_original.drop(columns=['p_airout'])
    test_df_combined = pd.concat([test_df_original.reset_index(drop=True), test_df[['p_airout']].reset_index(drop=True)], axis=1)
    
    # Save predictions
    save_predictions(test_df_combined, '../../data/Seattle Mariners 2025 Analytics Internship/test_predictions.csv', debug=True)

    print("Predictions made successfully and saved.")

if __name__ == '__main__':
    main()

