
def predict(model, X, debug=False):
    """
    Make predictions using the trained model.
    """
    if debug:
        print("Making predictions...")
    return model.predict_proba(X)[:, 1]  # Probability of class 1

def save_predictions(predictions, output_path, debug=False):
    """
    Save predictions to a CSV file.
    """
    predictions.to_csv(output_path, index=False)
    if debug:
        print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    import pandas as pd
    # from data_loading import load_data
    # from preprocessing import clean_data
    # from feature_engineering import feature_engineering_pipeline
    # from training_and_eval import load_model
    
    train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train.csv'
    test_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-test.csv'
    dict_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-dictionary.csv'
    
    # Load data with debug mode enabled
    train_df, test_df, data_dict_df = load_data(train_path, test_path, dict_path, debug=True)
    test_df = clean_data(test_df)
    test_df_original = test_df.copy() 
    test_df = feature_engineering_pipeline(test_df)
    features = test_df.columns  # Assuming the same features as training
    X_test = test_df[features]
    
    # Load model
    model = load_model('../../data/Seattle Mariners 2025 Analytics Internship/models/catboost_best.pkl')
    
    # Make predictions
    test_df['p_airout'] = predict(model, X_test, debug=True)

    # Combine predictions with original test data
    test_df_original = test_df_original.drop(columns=['p_airout'])
    test_df_combined = pd.concat([test_df_original.reset_index(drop=True), test_df[['p_airout']].reset_index(drop=True)], axis=1)
    
    # Save predictions
    save_predictions(test_df_combined, '../../data/Seattle Mariners 2025 Analytics Internship/test_predictions.csv', debug=True)
