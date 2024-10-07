
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_hit_trajectory(vert_exit_angle, horz_exit_angle, exit_speed, spin_rate, gravity=32.174, time_intervals=100):
    """
    Calculate the hit trajectory based on initial speed and angles.
    Args:
    - vert_exit_angle: Vertical exit angle in degrees.
    - horz_exit_angle: Horizontal exit angle in degrees.
    - exit_speed: Initial exit speed in mph.
    - spin_rate: Spin rate in rpm.
    - gravity: Gravity constant in ft/s^2.
    - time_intervals: Number of time intervals for calculating the trajectory.

    Returns:
    - x_values: List of x coordinates of the trajectory.
    - y_values: List of y coordinates of the trajectory.
    """
    # Convert exit speed to ft/s (1 mph = 1.46667 ft/s)
    exit_speed_ft_s = exit_speed * 1.46667

    # Convert angles to radians
    vert_angle_rad = np.radians(vert_exit_angle)
    horz_angle_rad = np.radians(horz_exit_angle)

    # Calculate initial velocity components
    vx = exit_speed_ft_s * np.cos(vert_angle_rad)
    vy = exit_speed_ft_s * np.sin(vert_angle_rad)

    # Calculate the effect of spin on horizontal and vertical distances
    # Approximate adjustment factor due to spin rate
    spin_effect = 1 + (spin_rate / 15000)

    # Calculate trajectory points
    time_points = np.linspace(0, 5, time_intervals)  # 5 seconds max trajectory time
    x_values = (vx * time_points) * np.cos(horz_angle_rad) * spin_effect
    y_values = (vy * time_points - 0.5 * gravity * time_points ** 2) * spin_effect

    # Remove any points where y is negative (ground level or below)
    valid_indices = y_values >= 0
    x_values = x_values[valid_indices]
    y_values = y_values[valid_indices]

    return x_values, y_values

def visualize_hit_trajectory(df, boundary=None, debug=False):
    """
    Visualize the hit's trajectory and landing point with respect to the field boundaries.
    Ensure that the boundary and trajectory visualization match the calculation methods.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Simulate outfield boundary using the same function and parameters as used for determining home runs
    if boundary is None:
        boundary = simulate_outfield_boundary(average_distances, debug=debug)

    boundary_x, boundary_y, _, _ = boundary

    # Visualize the outfield boundary
    ax.plot(boundary_x, boundary_y, color='black', linestyle='--', linewidth=2, label='Outfield Boundary')

    # Plot each hit's trajectory
    for idx, row in df.iterrows():
        # Calculate the trajectory using the initial speed, angles, and spin rate
        x_traj, y_traj = calculate_hit_trajectory(
            row['vert_exit_angle'],
            row['horz_exit_angle'],
            row['exit_speed'],
            row['hit_spin_rate']
        )

        # Plot trajectory line
        ax.plot(x_traj, y_traj, linestyle='-', alpha=0.6)

        # Plot the landing point
        ax.scatter(row['landing_x_adjusted'], row['landing_y_adjusted'], color='red', s=50, label='Landing Point' if idx == 0 else "")

    # Set field boundary limits to show the full field
    max_x = max(boundary_x) + 50
    max_y = max(boundary_y) + 50
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(0, max_y)

    # Formatting the plot
    ax.set_title("Hit Trajectory and Landing Points with Outfield Boundary")
    ax.set_xlabel("Landing X (ft)")
    ax.set_ylabel("Landing Y (ft)")
    ax.axhline(0, color='black', linewidth=1)  # Baseline for y-axis
    ax.axvline(0, color='black', linewidth=1)  # Baseline for x-axis
    ax.legend(title='Legend')
    ax.grid(True)

    return fig



#https://www.si.com/mlb/2021/03/24/mlb-outfield-walls-ranked-fenway-park-yankee-stadium

# Provided data
stadium_data = {
    'Kauffman Stadium': [330, 387, 410, 387, 330],
    'Rogers Centre': [328, 375, 400, 375, 328],
    'TD Ballpark': [333, 380, 400, 363, 336],
    'Busch Stadium': [336, 375, 400, 375, 335],
    'Dodger Stadium': [330, 360, 375, 400, 375, 360, 330],
    'Guaranteed Rate Field': [330, 375, 400, 375, 335],
    'Oakland Coliseum': [330, 388, 400, 388, 330],
    'Marlins Park': [344, 386, 400, 387, 335],
    'Miller Park': [344, 371, 400, 374, 345],
    'T-Mobile Park': [331, 378, 401, 381, 326],
    'Citi Field': [335, 358, 385, 408, 398, 375, 330],
    'Tropicana Field': [315, 370, 404, 370, 322],
    'Truist Park': [335, 385, 400, 375, 325],
    'Wrigley Field': [355, 368, 400, 368, 353],
    'Coors Field': [347, 390, 415, 375, 350],
    'Angel Stadium': [347, 390, 396, 370, 365, 350],
    'Comerica Park': [345, 370, 420, 365, 330],
    'Great American Ball Park': [328, 379, 404, 370, 325],
    'Nationals Park': [337, 377, 402, 370, 335],
    'Progressive Field': [325, 370, 400, 410, 375, 325],
    'Target Field': [339, 377, 411, 403, 367, 328],
    'Oriole Park at Camden Yards': [333, 364, 410, 400, 373, 318],
    'Chase Field': [330, 374, 413, 407, 413, 374, 334],
    'Globe Life Field': [329, 372, 407, 374, 326],
    'Petco Park': [334, 357, 390, 396, 391, 382, 322],
    'Citizens Bank Park': [329, 374, 409, 401, 369, 330],
    'Yankee Stadium': [318, 399, 408, 385, 314],
    'PNC Park': [325, 383, 410, 399, 375, 320],
    'Minute Maid Park': [315, 362, 404, 409, 408, 373, 326],
    'Oracle Park': [339, 364, 399, 391, 415, 365, 309],
    'Fenway Park': [310, 379, 390, 420, 380, 302]
}

# Initialize an empty list to collect data
data = []

for stadium, distances in stadium_data.items():
    num_points = len(distances)
    if num_points == 5:
        # Map directly
        P1 = distances[0]  # Left Field Line
        P2 = distances[1]  # Left-Center Field
        P3 = distances[2]  # Center Field
        P4 = distances[3]  # Right-Center Field
        P5 = distances[4]  # Right Field Line
    elif num_points == 6:
        # Use positions 0,1,2,3,5
        P1 = distances[0]
        P2 = distances[1]
        P3 = distances[2]
        P4 = distances[3]
        P5 = distances[5]
    elif num_points == 7:
        # Use positions 0,2,3,4,6
        P1 = distances[0]
        P2 = distances[2]
        P3 = distances[3]
        P4 = distances[4]
        P5 = distances[6]
    else:
        # Handle other cases if necessary
        continue  # Skip if the number of distances is not 5,6,7

    data.append({
        'Stadium': stadium,
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'P4': P4,
        'P5': P5
    })

# Create DataFrame
df = pd.DataFrame(data)

# Calculate averages
average_P1 = df['P1'].mean()
average_P2 = df['P2'].mean()
average_P3 = df['P3'].mean()
average_P4 = df['P4'].mean()
average_P5 = df['P5'].mean()

# Display the DataFrame and averages
print("Stadium Distances DataFrame:")
print(df.to_string(index=False))

print("\nAverage Distances:")
print(f"Left Field Line (P1): {average_P1:.2f} ft")
print(f"Left-Center Field (P2): {average_P2:.2f} ft")
print(f"Center Field (P3): {average_P3:.2f} ft")
print(f"Right-Center Field (P4): {average_P4:.2f} ft")
print(f"Right Field Line (P5): {average_P5:.2f} ft")



def categorize_inning(df, debug=False):
    if debug:
        print(f"Columns before categorizing innings: {df.columns.tolist()}")
        print(f"Checking if 'inning' exists in the DataFrame: {'inning' in df.columns}")
        if 'inning' not in df.columns:
            print("ERROR: 'inning' column is missing. Exiting function early.")
            return df  # Exit early if the 'inning' column is missing

    # Proceed to categorize 'inning' only if it exists
    if 'inning' in df.columns:
        df['inning_group'] = pd.cut(df['inning'], bins=[0, 3, 6, np.inf], labels=['Early', 'Mid', 'Late'])
        df = df.drop(columns=['inning'])  # Drop the original inning column if not needed
    if debug:
        print(f"Columns after categorizing innings: {df.columns.tolist()}")
    return df

def create_count_scenario(df, debug=False):
    """
    Combine 'pre_balls', 'pre_strikes', 'pre_outs', and 'inning_group' into 'count_scenario'.
    """
    if debug:
        print(f"Columns before create_count_scenario: {df.columns.tolist()}")
    # Check for required columns before proceeding
    required_columns = ['pre_balls', 'pre_strikes', 'pre_outs', 'inning_group']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}. Available columns: {df.columns.tolist()}")
    
    # Combine game context features into a single string representation
    df['count_scenario'] = (df['pre_balls'].astype(str) + '-' + 
                            df['pre_strikes'].astype(str) + '-' +
                            df['pre_outs'].astype(str) + '-' +
                            df['inning_group'].astype(str))
    
    # Drop the original columns if not needed anymore
    df = df.drop(columns=['pre_balls', 'pre_strikes', 'pre_outs', 'inning_group'])
    
    if debug:
        print("Unique 'count_scenario's:", df['count_scenario'].unique())
    
    return df

def transform_gamedate(df, debug=False):
    if debug:
        print("Transforming 'gamedate' column...")
    df['gamedate'] = pd.to_datetime(df['gamedate'])
    df['month'] = df['gamedate'].dt.month
    df['day_of_week'] = df['gamedate'].dt.day_name()
    df['is_weekend'] = df['gamedate'].dt.dayofweek >= 5
    df = df.drop(columns=['gamedate'])
    if debug:
        print("Transformed columns:", df.columns)
    return df

def categorize_temperature(df, debug=False):
    if debug:
        print("Categorizing 'temperature' column...")
    bins = [0, 70, 90, np.inf]
    labels = ['Cold', 'Moderate', 'Hot']
    df['temperature_category'] = pd.cut(df['temperature'], bins=bins, labels=labels)
    df = df.drop(columns=['temperature'])
    if debug:
        print("Temperature categories assigned:", df['temperature_category'].unique())
    return df

def calculate_physics_features(df, debug=False):
    if debug:
        print("Calculating physics-based features...")
    GRAVITY = 32.174  # ft/s^2
    df['vert_angle_rad'] = np.radians(df['vert_exit_angle'])
    df['estimated_distance'] = ((df['exit_speed'] ** 2) * np.sin(2 * df['vert_angle_rad'])) / GRAVITY
    df['landing_x'] = df['estimated_distance'] * np.sin(np.radians(df['horz_exit_angle']))
    df['landing_y'] = df['estimated_distance'] * np.cos(np.radians(df['horz_exit_angle']))
    df['adjusted_distance'] = df['estimated_distance'] * (1 + (df['hit_spin_rate'] / 15000))
    df['landing_x_adjusted'] = df['adjusted_distance'] * np.sin(np.radians(df['horz_exit_angle']))
    df['landing_y_adjusted'] = df['adjusted_distance'] * np.cos(np.radians(df['horz_exit_angle']))
    
    if debug:
        print("Physics features calculated.")
        print(df[['estimated_distance', 'adjusted_distance', 'landing_x_adjusted', 'landing_y_adjusted']].head())
    
    return df

def determine_home_run(df, boundary=None, debug=False):
    """
    Determine if each hit in the DataFrame is a home run using the precomputed outfield boundary.
    """
    if boundary is None:
        boundary = simulate_outfield_boundary(average_distances, debug=False)  # Use precomputed boundary

    boundary_x, boundary_y, _, _ = boundary

    def is_home_run(row):
        x = row['landing_x_adjusted']
        y = row['landing_y_adjusted']
        distance = np.sqrt(x**2 + y**2)

        # Calculate the angle of the hit
        hit_angle = np.degrees(np.arctan2(x, y))

        # Find the corresponding boundary distance for this angle
        if -45 <= hit_angle <= 45:
            # Find the closest point in the boundary array
            idx = np.abs(np.linspace(-45, 45, 100) - hit_angle).argmin()
            boundary_distance = np.sqrt(boundary_x[idx]**2 + boundary_y[idx]**2)
            return distance >= boundary_distance

        return False

    df['is_home_run'] = df.apply(is_home_run, axis=1)
    if debug:
        print("Number of home runs:", df['is_home_run'].sum())
        print(df[df['is_home_run'] == True][['hit_direction', 'adjusted_distance']].head())

    return df

global_boundary = None

average_distances = {
    'P1': 332.45,  # Average Left Field Line
    'P2': 381.55,  # Average Left-Center Field
    'P3': 403.48,  # Average Center Field
    'P4': 385.81,  # Average Right-Center Field
    'P5': 329.16   # Average Right Field Line
}

# Updated function to simulate outfield boundary with debug outputs at every 5 feet
def simulate_outfield_boundary(average_distances, debug=False):
    """
    Create a gradual outfield boundary using a polynomial curve fit or similar, using average distances.
    Output dimensions once for debugging purposes if debug is True.
    """
    global global_boundary  # Use a global boundary to avoid repeated calculations
    
    # If the boundary is already calculated, return it directly.
    if global_boundary is not None:
        return global_boundary

    # Extract average distances for each field position
    P1 = average_distances['P1']
    P2 = average_distances['P2']
    P3 = average_distances['P3']
    P4 = average_distances['P4']
    P5 = average_distances['P5']

    # Define angles corresponding to each point
    angles = np.linspace(-45, 45, 100)  # Covering left to right field (in degrees)
    angles_rad = np.radians(angles)

    # Calculate polynomial coefficients based on these points
    # Create a smooth curve that fits through (Left Field Line, Left-Center, Center Field, Right-Center, Right Field Line)
    boundary_coefficients = np.polyfit(
        [-45, -22.5, 0, 22.5, 45],  # angles corresponding to each average point
        [P1, P2, P3, P4, P5],  # distance values
        deg=3  # Cubic polynomial fit
    )

    # Generate boundary distances using the fitted polynomial
    boundary_distances = np.polyval(boundary_coefficients, angles)

    # Calculate boundary x and y positions based on these distances
    boundary_x = boundary_distances * np.sin(angles_rad)
    boundary_y = boundary_distances * np.cos(angles_rad)

    # Generate outfield boundary points every 5 feet for debugging purposes
    distances_5ft = np.arange(0, max(boundary_distances), 5)
    boundary_x_5ft = []
    boundary_y_5ft = []

    # Populate boundary values at every 5-foot interval for the outfield
    for d in distances_5ft:
        # Find the corresponding x and y for this distance
        idx = (np.abs(boundary_distances - d)).argmin()
        boundary_x_5ft.append(boundary_x[idx])
        boundary_y_5ft.append(boundary_y[idx])

        if debug:
            print(f"Outfield Boundary at {d} ft: (x={boundary_x[idx]:.2f}, y={boundary_y[idx]:.2f})")

    # Store the calculated boundary in the global variable
    global_boundary = (boundary_x, boundary_y, boundary_x_5ft, boundary_y_5ft)

    return global_boundary


def visualize_in_park_foul_balls(df, title='In-Park Foul Balls'):
    """
    Visualize the in-park foul balls that are catchable.
    """
    plt.figure(figsize=(12, 10))
    in_park_foul_df = df[df['is_foul'] & ~df['is_home_run']]  # Filter only catchable in-park foul balls
    
    sns.scatterplot(x='landing_x_adjusted', y='landing_y_adjusted', hue='hit_direction', data=in_park_foul_df, palette='deep')
    
    # Additional visual aids
    plt.axhline(0, color='black', linewidth=1)  # Baseline for y-axis
    plt.axvline(0, color='black', linewidth=1)  # Baseline for x-axis
    plt.title(title)
    plt.xlabel('Landing X (ft)')
    plt.ylabel('Landing Y (ft)')
    plt.legend(title='Hit Direction')
    plt.grid()
    plt.show()
    
def visualize_hits_with_field_boundary(df, title='Hit Landing Points', hits_type='all', debug=False):
    """
    Visualize the landing points of hits with different categories, including the outfield boundary.
    """
    # Generate the field boundary
    boundary_x, boundary_y, _, _ = simulate_outfield_boundary(average_distances, debug=True)
    
    plt.figure(figsize=(12, 10))
    
    # Determine the data to plot based on the hit type
    if hits_type == 'all':
        data = df
    elif hits_type == 'home_runs':
        data = df[df['is_home_run']]
    elif hits_type == 'foul_balls':
        data = df[df['is_foul']]
    else:
        data = df

    # Plot the hit points
    sns.scatterplot(x='landing_x_adjusted', y='landing_y_adjusted', hue='hit_direction', data=data, palette='deep')

    # Plot the field boundary
    plt.plot(boundary_x, boundary_y, color='black', linestyle='--', linewidth=2, label='Outfield Boundary')
    
    # Additional visual aids
    plt.axhline(0, color='black', linewidth=1)  # Baseline for y-axis
    plt.axvline(0, color='black', linewidth=1)  # Baseline for x-axis
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel('Landing X (ft)')
    plt.ylabel('Landing Y (ft)')
    plt.legend(title='Hit Direction')
    plt.grid()
    plt.show()

# Modify is_within_field_boundaries to use the simulated boundary curve
def is_within_field_boundaries(row, boundary=None, debug=False):
    """
    Check if the ball lands within the field boundaries using the precomputed boundary.
    """
    if boundary is None:
        boundary = simulate_outfield_boundary(average_distances, debug=False)  # Use precomputed boundary

    boundary_x, boundary_y, boundary_x_5ft, boundary_y_5ft = boundary

    x = row['landing_x_adjusted']
    y = row['landing_y_adjusted']

    # Determine if point is inside the boundary curve by checking the distance to the origin
    distance = np.sqrt(x**2 + y**2)

    # Calculate the angle of the hit
    hit_angle = np.degrees(np.arctan2(x, y))

    # Find the corresponding boundary distance for this angle
    if -45 <= hit_angle <= 45:
        # Find the closest point in the boundary array
        idx = np.abs(np.linspace(-45, 45, 100) - hit_angle).argmin()
        boundary_distance = np.sqrt(boundary_x[idx]**2 + boundary_y[idx]**2)
        return distance <= boundary_distance

    return False

def determine_catchable_home_run(df, debug=False):
    if debug:
        print("Determining catchable home runs...")
    def is_catchable(row):
        if row['is_home_run'] and row['exit_speed'] < 110 and row['hit_spin_rate'] < 3500 and is_within_field_boundaries(row):
            return True
        return False
    df['is_catchable_home_run'] = df.apply(is_catchable, axis=1)
    if debug:
        print("Number of catchable home runs:", df['is_catchable_home_run'].sum())
        print(df[df['is_catchable_home_run'] == True][['hit_direction', 'adjusted_distance', 'landing_x_adjusted', 'landing_y_adjusted']].head())
    return df

def categorize_hit_direction(df, debug=False):
    """
    Categorize hits into 'Left', 'Center', 'Right' based on 'horz_exit_angle'.
    """
    if debug:
        print("Categorizing hit directions...")
    
    conditions = [
        df['horz_exit_angle'] < -15,
        df['horz_exit_angle'] > 15
    ]
    choices = ['Left', 'Right']
    df['hit_direction'] = np.select(conditions, choices, default='Center')
    
    if debug:
        print("Hit directions assigned:", df['hit_direction'].unique())
    
    return df

def filter_features(df, selected_features, include_target=False, target_variable='is_airout', debug=False):
    """
    Filter the DataFrame to only include columns specified in selected_features,
    plus the target variable if include_target is True.
    """
    if debug:
        print(f"Original columns: {df.columns.tolist()}")
    
    # Prepare the list of columns to keep
    columns_to_keep = selected_features.copy()
    if include_target and target_variable in df.columns:
        columns_to_keep.append(target_variable)
    
    # Filter columns
    df_filtered = df[columns_to_keep].copy()

    if debug:
        print(f"Filtered columns: {df_filtered.columns.tolist()}")
    
    return df_filtered


def visualize_hits(df, title='Hit Landing Points'):
    """
    Visualize the landing points of hits with different categories.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='landing_x_adjusted', y='landing_y_adjusted', hue='hit_direction', style='is_home_run', data=df, palette='deep')
    plt.axhline(0, color='black', linewidth=1)  # Baseline for y-axis
    plt.axvline(0, color='black', linewidth=1)  # Baseline for x-axis
    plt.title(title)
    plt.xlabel('Landing X (ft)')
    plt.ylabel('Landing Y (ft)')
    plt.legend(title='Hit Direction / Home Run')
    plt.grid()
    plt.show()

    # Show catchable home runs
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='landing_x_adjusted', y='landing_y_adjusted', hue='is_catchable_home_run', data=df, palette={True: 'green', False: 'red'})
    plt.title(f'{title} - Catchable Home Runs')
    plt.xlabel('Landing X (ft)')
    plt.ylabel('Landing Y (ft)')
    plt.legend(title='Catchable Home Run')
    plt.grid()
    plt.show()

# Updated feature engineering pipeline to include pre-filtering visualizations
def feature_engineering_pipeline(df, selected_features=None, include_target=False, debug=False):
    """
    Run all feature engineering functions and filter for selected features if provided.
    """
    if debug:
        print(f"Initial columns: {df.columns.tolist()}")
        
    global global_boundary  # Use global boundary to ensure consistency
    if global_boundary is None:
        global_boundary = simulate_outfield_boundary(average_distances, debug=debug)  # Compute once if not set
    df = categorize_inning(df, debug)
    df = create_count_scenario(df, debug)
    df = transform_gamedate(df, debug)
    df = categorize_temperature(df, debug)
    df = calculate_physics_features(df, debug)
    df = categorize_hit_direction(df, debug)
    df = determine_home_run(df, boundary=None, debug=debug)
    df = determine_catchable_home_run(df, debug=debug)

    # Create 'is_foul' before the first visualization
    if debug:
        print("Determining foul balls...")
    df['is_foul'] = ~df.apply(lambda row: is_within_field_boundaries(row, boundary=global_boundary), axis=1)

    # Visualize before filtering
    print("Visualizing hits before applying any filters...")
    visualize_hits_with_field_boundary(df, title='All Hit Landing Points Before Filtering', hits_type='all', debug=debug)

    print("Visualizing home runs before filtering...")
    visualize_hits_with_field_boundary(df[df['is_home_run']], title='Home Runs Before Filtering', hits_type='home_runs', debug=debug)

    print("Visualizing foul balls before filtering...")
    visualize_hits_with_field_boundary(df[df['is_foul']], title='Foul Balls Before Filtering', hits_type='foul_balls', debug=debug)

    # Apply filtering for non-catchable and foul balls
    df = filter_non_catchable_and_foul(df, debug)

    # Visualize after filtering
    print("Visualizing hits after filtering uncatchable balls...")
    visualize_hits_with_field_boundary(df, title='All Hit Landing Points After Filtering', hits_type='all', debug=debug)

    print("Visualizing home runs after filtering...")
    visualize_hits_with_field_boundary(df[df['is_home_run']], title='Home Runs After Filtering', hits_type='home_runs', debug=debug)

    print("Visualizing foul balls after filtering...")
    visualize_hits_with_field_boundary(df[df['is_foul']], title='Foul Balls After Filtering', hits_type='foul_balls', debug=debug)

    # Drop intermediate columns if necessary
    df = df.drop(columns=['vert_angle_rad'], errors='ignore')

    # Filter the DataFrame for selected features
    if selected_features:
        df = filter_features(df, selected_features, include_target=include_target, debug=debug)

    return df

# Updated filter_non_catchable_and_foul function to clarify debug prints and ensure visibility
def filter_non_catchable_and_foul(df, debug=False):
    if debug:
        print("Filtering non-catchable and foul balls...")
        total_rows_before = df.shape[0]
        
    # Determine if each hit is a foul ball or not within field boundaries
    df['is_foul'] = ~df.apply(is_within_field_boundaries, axis=1)
    
    # Filter to keep only catchable home runs or hits that are not fouls (remain in the park)
    df_filtered = df[df['is_catchable_home_run'] | ~df['is_foul']]
    
    if debug:
        total_rows_after = df_filtered.shape[0]
        print(f"Rows before filtering: {total_rows_before}")
        print(f"Rows after filtering: {total_rows_after}")
        print(f"Number of rows filtered out: {total_rows_before - total_rows_after}")
        print("Filtered out rows (foul balls outside park):")
        print(df[df['is_foul'] & ~df['is_catchable_home_run']][['hit_direction', 'adjusted_distance', 'landing_x_adjusted', 'landing_y_adjusted']].head())
    
    return df_filtered

# Run the updated pipeline
if __name__ == '__main__':
    import pandas as pd
    # from data_loading import load_data
    # from preprocessing import clean_data, handle_missing_values
    
    train_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-train.csv'
    test_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-test.csv'
    dict_path = '../../data/Seattle Mariners 2025 Analytics Internship/data-dictionary.csv'

    # Load data with debug mode enabled
    train_df, test_df, data_dict_df = load_data(train_path, test_path, dict_path, debug=True)

    # Preprocess data
    train_df = clean_data(train_df, debug=True)
    train_df = handle_missing_values(train_df, debug=True)

    # Apply feature engineering with debug mode enabled
    train_df = feature_engineering_pipeline(train_df, debug=True)
