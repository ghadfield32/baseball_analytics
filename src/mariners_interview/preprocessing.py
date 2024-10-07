
def clean_data(df, debug=False):
    if debug:
        print(f"DataFrame shape after loading data: {df.shape}")
        print(f"Columns before cleaning: {df.columns.tolist()}")
    # Assuming the cleaning process doesn't drop the 'inning' column
    if debug:
        print(f"Columns after cleaning: {df.columns.tolist()}")
    return df

def handle_missing_values(df, debug=False):
    if debug:
        print(f"Columns before handling missing values: {df.columns.tolist()}")
    # Example of a missing value handling process
    df = df[df['hit_spin_rate'].notnull()]  # Assuming hit_spin_rate is critical and should not have nulls

    if debug:
        print(f"Columns after handling missing values: {df.columns.tolist()}")
    return df

if __name__ == '__main__':
    # import pandas as pd
    # from data_loading import load_data
    
    # Load data for testing
    train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train.csv'
    test_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-test.csv'
    dict_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-dictionary.csv'
    
    # Load data with debug mode enabled
    train_df, test_df, data_dict_df = load_data(train_path, test_path, dict_path, debug=True)
    
    # Clean data with debug mode enabled
    train_df = clean_data(train_df, debug=True)
    train_df = handle_missing_values(train_df, debug=True)
