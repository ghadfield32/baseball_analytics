

import numpy as np
import pandas as pd

# Constants
GRAVITY = 32.174  # Gravity constant in ft/s^2

# Function to calculate hang time based on exit speed and vertical angle
def calculate_hang_time(df, debug=False):
    if debug:
        print("\nCalculating hang time for each hit...")
    
    # Convert vertical exit angle to radians for trigonometric calculations
    df['vert_angle_rad'] = np.radians(df['vert_exit_angle'])
    
    # Hang time calculation using kinematic equation: t = (2 * exit_speed * sin(vert_exit_angle)) / GRAVITY
    df['hang_time'] = (2 * df['exit_speed'] * np.sin(df['vert_angle_rad'])) / GRAVITY
    
    # Ensure no negative or zero hang times
    df['hang_time'] = df['hang_time'].clip(lower=0)

    if debug:
        # Show the first few hang time calculations
        print("Hang time calculated for the first few records:\n", df[['hang_time', 'exit_speed', 'vert_exit_angle']].head())
    return df

# Function to estimate distance covered by the fielder based on their starting position
def calculate_distance_covered(df, debug=False):
    if debug:
        print("\nEstimating distance covered by fielder...")

    # Define starting positions for each outfielder based on standard field layout
    lf_start_x, lf_start_y = -90, 250  # Left Fielder starting position
    cf_start_x, cf_start_y = 0, 350    # Center Fielder starting position
    rf_start_x, rf_start_y = 90, 250   # Right Fielder starting position

    # Function to calculate the distance from the starting position to the landing position
    def get_distance_covered(row):
        # Select the starting position based on fielder role
        if row['first_fielder'] == row['lf_id']:
            start_x, start_y = lf_start_x, lf_start_y
        elif row['first_fielder'] == row['cf_id']:
            start_x, start_y = cf_start_x, cf_start_y
        elif row['first_fielder'] == row['rf_id']:
            start_x, start_y = rf_start_x, rf_start_y
        else:
            return np.nan  # Return NaN if no matching fielder found
        
        # Calculate distance covered from starting position to landing position
        distance = np.sqrt((row['landing_x_adjusted'] - start_x) ** 2 + (row['landing_y_adjusted'] - start_y) ** 2)
        return distance

    # Apply distance calculation to each row
    df['distance_covered'] = df.apply(get_distance_covered, axis=1)

    if debug:
        # Display intermediate distances and validate with initial starting positions
        print("Distance covered calculated for the first few records:\n", df[['distance_covered', 'landing_x_adjusted', 'landing_y_adjusted']].head())
    return df

# Function to estimate fielder reaction speed
def calculate_reaction_speed(df, debug=False):
    if debug:
        print("\nCalculating fielder reaction speed...")

    # Reaction speed is calculated as distance covered divided by hang time
    df['reaction_speed'] = df['distance_covered'] / df['hang_time']
    
    # Replace infinite or NaN values with zero for reaction speed
    df['reaction_speed'] = df['reaction_speed'].replace([np.inf, -np.inf], 0).fillna(0)

    if debug:
        # Validate the reaction speed for the first few records
        print("Reaction speed calculated for the first few records:\n", df[['reaction_speed', 'distance_covered', 'hang_time']].head())
    return df

# Function to calculate estimated catch probability
def calculate_catch_probability(df, debug=False):
    if debug:
        print("\nEstimating catch probability...")

    # Estimate catch probability using an exponential decay model based on distance covered and hang time
    df['catch_probability'] = np.exp(-0.05 * df['distance_covered']) * (df['hang_time'] / 5)

    # Clip catch probability to be between 0 and 1
    df['catch_probability'] = np.clip(df['catch_probability'], 0, 1)

    if debug:
        # Display catch probability and related calculations
        print("Catch probability estimated for the first few records:\n", df[['catch_probability', 'distance_covered', 'hang_time']].head())
    return df

# Function to categorize catch difficulty based on distance and hang time
def categorize_catch_difficulty(df, debug=False):
    if debug:
        print("\nCategorizing catch difficulty...")

    # Step 1: Initialize catch_difficulty to 'Not_Caught' where is_airout is 0
    df['catch_difficulty'] = np.where(df['is_airout'] == 0, 'Not_Caught', 'Unknown')

    # Step 2: Define categorization conditions based on available metrics for other rows
    conditions = [
        (df['distance_covered'] < 50) & (df['hang_time'] > 4),             # Short distance, high hang time = Easy catch
        (df['distance_covered'] < 100) & (df['hang_time'] > 3),            # Medium distance, moderate hang time = Moderate catch
        (df['distance_covered'] >= 100) & (df['hang_time'] < 3),           # Long distance, low hang time = Difficult catch
        (df['distance_covered'] >= 150) | (df['hang_time'] < 2)            # Very long distance or very low hang time = Very Difficult catch
    ]
    choices = ['Easy', 'Moderate', 'Difficult', 'Very Difficult']

    # Step 3: Calculate catch difficulty for all rows using np.select
    df['temp_catch_difficulty'] = np.select(conditions, choices, default='Uncatchable')

    # Step 4: Overwrite catch_difficulty only for rows where valid metrics exist
    mask_valid_metrics = ~df['first_fielder'].isna() | (~df['distance_covered'].isna() & ~df['hang_time'].isna())
    df.loc[mask_valid_metrics, 'catch_difficulty'] = df.loc[mask_valid_metrics, 'temp_catch_difficulty']

    # Drop the temporary column used for calculation
    df.drop(columns=['temp_catch_difficulty'], inplace=True)

    if debug:
        # Show categorized difficulty for initial records
        print("Catch difficulty categorized:\n", df[['catch_difficulty', 'distance_covered', 'hang_time', 'first_fielder', 'is_airout']].head())
    
    return df

# Function to determine the position of the fielder who attempted the catch
def determine_fielder_position(row):
    if row['first_fielder'] == row['lf_id']:
        return 'LF'
    elif row['first_fielder'] == row['cf_id']:
        return 'CF'
    elif row['first_fielder'] == row['rf_id']:
        return 'RF'
    return 'Unknown'  # Catch not made or missing fielder ID information

# Function to get count of unique values for catch difficulty
def get_catch_difficulty_count(df, debug=False):
    if debug:
        print("\nGetting catch difficulty counts...")
    
    # Get counts for each unique value in the catch_difficulty column
    catch_difficulty_counts = df['catch_difficulty'].value_counts()
    if debug:
        print("Catch difficulty counts:\n", catch_difficulty_counts)
    return catch_difficulty_counts

# Extend the existing feature engineering pipeline with these new metrics
def feature_engineering_with_defensive_metrics(df, selected_features=None, include_target=False, debug=False):

    # Calculate new metrics for defensive evaluation
    df = calculate_hang_time(df, debug)
    df = calculate_distance_covered(df, debug)
    df = calculate_reaction_speed(df, debug)
    df = calculate_catch_probability(df, debug)
    df = categorize_catch_difficulty(df, debug)
    df['fielder_position'] = df.apply(determine_fielder_position, axis=1)

    return df

# Example use of the pipeline
if __name__ == '__main__':
    # Assuming train_df is already loaded and preprocessed
    train_df = feature_engineering_with_defensive_metrics(train_df, debug=True)
    print("train_df columns =", train_df.columns)
    
    # Get and print catch difficulty counts
    catch_difficulty_counts = get_catch_difficulty_count(train_df, debug=True)


# defensive metrics cause data leakage, great for post prediction analsis on these players
