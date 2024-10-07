
import pandas as pd

def load_data(train_path, test_path, dict_path, debug=False):
    """
    Load the training and testing datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    data_dict_df = pd.read_csv(dict_path)
    
    if debug:
        print("Train Data Shape:", train_df.head())
        print("Test Data Shape:", test_df.head())
        print("Data Dictionary Shape:", data_dict_df)
    
    return train_df, test_df, data_dict_df

if __name__ == '__main__':
    # Paths to data files
    train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train.csv'
    test_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-test.csv'
    dict_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-dictionary.csv'
    
    # Load data with debug mode enabled
    train_df, test_df, data_dict_df = load_data(train_path, test_path, dict_path, debug=True)
