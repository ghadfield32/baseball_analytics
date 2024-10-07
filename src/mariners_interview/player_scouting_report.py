

# Import necessary libraries
import pandas as pd
import numpy as np

# Step 1: Load and preprocess the dataset
def load_data(file_path, player_id=None, debug=False):
    """
    Load the dataset and filter it to only include the given player's data.
    Additional preprocessing includes filtering out 'Uncatchable' from 'catch_difficulty'.
    """
    league_df = pd.read_csv(file_path)
    
    # Remove "Uncatchable" entries
    league_df = league_df[league_df['catch_difficulty'] != 'Uncatchable']

    # Convert numeric columns to appropriate types
    numeric_columns = league_df.select_dtypes(include=['float64', 'int64']).columns
    league_df[numeric_columns] = league_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Filter for the given player
    player_df = league_df[league_df['cf_id'] == player_id] if player_id else None

    if debug:
        print("Loaded DataFrame Shape: ", league_df.shape)
        print("Filtered Player Data Shape: ", player_df.shape if player_df is not None else 'No Player Filter')
        print("Columns in Dataset: ", league_df.columns)
    
    return player_df, league_df

# Step 2: Calculate percentile ranks
def calculate_percentiles(df, metrics):
    percentiles_df = pd.DataFrame(index=df.index)
    for metric in metrics:
        percentiles_df[f'{metric}_Percentile'] = df[metric].rank(pct=True) * 100
    return percentiles_df

# Step 3: Compare player statistics to league averages
def compare_player_to_league(player_df, league_df, metrics, debug=False):
    player_avg = player_df[metrics].mean()
    league_avg = league_df[metrics].mean()
    
    comparison_df = pd.DataFrame({'Player Average': player_avg})
    comparison_df['League Average'] = league_avg
    comparison_df['Difference'] = comparison_df['Player Average'] - comparison_df['League Average']
    comparison_df['Percent Difference'] = ((comparison_df['Player Average'] - comparison_df['League Average']) / comparison_df['League Average']) * 100

    player_percentiles = calculate_percentiles(player_df, metrics).mean()
    league_percentiles = calculate_percentiles(league_df, metrics).mean()

    for metric in metrics:
        comparison_df[f'Player {metric} Percentile'] = player_percentiles[f'{metric}_Percentile']
        comparison_df[f'League {metric} Percentile'] = league_percentiles[f'{metric}_Percentile']

    if debug:
        print("Player vs League Comparison with Percentiles:\n", comparison_df)
    
    return comparison_df

# Step 4: Compare player performance within cluster
def compare_player_to_cluster(player_df, league_df, cluster_column='defensive_cluster_label', metrics=None, debug=False):
    player_cluster = player_df[cluster_column].iloc[0]
    cluster_df = league_df[league_df[cluster_column] == player_cluster]
    
    if not metrics:
        metrics = cluster_df.select_dtypes(include=np.number).columns.tolist()

    cluster_summary = cluster_df[metrics].mean().to_frame(name='Cluster Average')
    player_summary = player_df[metrics].mean().to_frame(name='Player Average')

    cluster_comparison_df = pd.merge(player_summary, cluster_summary, left_index=True, right_index=True)
    cluster_comparison_df['Difference'] = cluster_comparison_df['Player Average'] - cluster_comparison_df['Cluster Average']
    cluster_comparison_df['Percent Difference'] = ((cluster_comparison_df['Player Average'] - cluster_comparison_df['Cluster Average']) / cluster_comparison_df['Cluster Average']) * 100

    player_percentiles = calculate_percentiles(player_df, metrics).mean()
    cluster_percentiles = calculate_percentiles(cluster_df, metrics).mean()

    for metric in metrics:
        cluster_comparison_df[f'Player {metric} Percentile'] = player_percentiles[f'{metric}_Percentile']
        cluster_comparison_df[f'Cluster {metric} Percentile'] = cluster_percentiles[f'{metric}_Percentile']

    if debug:
        print("Player vs Cluster Comparison with Percentiles:\n", cluster_comparison_df)

    return cluster_comparison_df

# Step : Compare player performance under different conditions
def compare_player_under_conditions(player_df, league_df, condition_column, metrics):
    """
    Compare player statistics under different conditions, such as temperature, venue, pitch side, etc.
    """
    # Get unique conditions
    unique_conditions = league_df[condition_column].unique()
    comparison_results = {}

    for condition in unique_conditions:
        # Filter data based on the current condition
        player_condition_df = player_df[player_df[condition_column] == condition]
        league_condition_df = league_df[league_df[condition_column] == condition]

        # Calculate the comparison between the player and league under this condition
        if not player_condition_df.empty and not league_condition_df.empty:
            comparison_df = compare_player_to_league(player_condition_df, league_condition_df, metrics)
            comparison_results[condition] = comparison_df

    return comparison_results


# Step 5: Generate Top and Bottom Percentiles Report
def get_top_and_bottom_percentiles(df, metrics, top_n=5, bottom_n=5, cluster_filter=None, catch_difficulty_filter=None):
    if cluster_filter:
        if isinstance(cluster_filter, list):
            df = df[df['defensive_cluster_label'].isin(cluster_filter)]
        else:
            df = df[df['defensive_cluster_label'] == cluster_filter]

    if catch_difficulty_filter:
        if isinstance(catch_difficulty_filter, list):
            df = df[df['catch_difficulty'].isin(catch_difficulty_filter)]
        else:
            df = df[df['catch_difficulty'] == catch_difficulty_filter]

    top_players = {}
    bottom_players = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"Metric {metric} not found in the DataFrame. Skipping...")
            continue
        
        top_players[metric] = df.nlargest(top_n, metric)
        bottom_players[metric] = df.nsmallest(bottom_n, metric)

    top_df = pd.concat(top_players, axis=0)
    bottom_df = pd.concat(bottom_players, axis=0)

    return top_df, bottom_df

# Step 6: Generate Scouting Report with Condition-Based Comparisons
def generate_scouting_report(file_path, player_id, metrics=None, condition_columns=None, debug=False):
    if metrics is None:
        metrics = ['reaction_speed', 'distance_covered', 'catch_probability']

    player_data, league_data = load_data(file_path, player_id, debug)
    league_comparison = compare_player_to_league(player_data, league_data, metrics, debug)

    report = f"""
    **Scouting Report: Player ID {player_id}**
    
    **Player Overview**:
    This report focuses on evaluating the defensive performance of the player (ID: {player_id}) based on their clustering, league comparison, and advanced metrics. The key metrics used are {', '.join(metrics)}.

    **League Comparison Analysis**:
    Against the league average, the player's performance is:
    {league_comparison.to_string()}
    """

    if condition_columns:
        report += "\n\n**Condition-Based Performance Analysis**:\n"
        for condition in condition_columns:
            condition_comparison = compare_player_under_conditions(player_data, league_data, condition, metrics)

            for condition_value, comparison_df in condition_comparison.items():
                report += f"\n**Performance under {condition}: {condition_value}**\n"
                for metric in metrics:
                    report += f"- {metric.capitalize()}: Player Average: {comparison_df.loc[metric, 'Player Average']:.2f}, League Average: {comparison_df.loc[metric, 'League Average']:.2f}, Difference: {comparison_df.loc[metric, 'Difference']:.2f}, Percent Difference: {comparison_df.loc[metric, 'Percent Difference']:.2f}%\n"
                
    return report, league_comparison

# Example usage with specific conditions and automated metrics
def main():
    file_path = "../../data/Seattle Mariners 2025 Analytics Internship/data-train-preprocessed.csv"
    player_id = 15411
    metrics_to_compare = ['reaction_speed', 'distance_covered', 'catch_probability']
    condition_columns = ['temperature_category', 'bat_side', 'pitch_side', 'top', 'venue_id', 'count_scenario']

# Generate and display report
    report, league_comparison = generate_scouting_report(file_path, player_id, metrics=metrics_to_compare, condition_columns=condition_columns, debug=True)
    print(report)

if __name__ == "__main__":
    main()
