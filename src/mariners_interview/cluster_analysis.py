
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
import streamlit as st

# Define a function to create a preprocessor for numeric features
def get_numeric_preprocessor(numeric_features, debug=False):
    """
    Create a preprocessor for handling numerical features.
    """
    if debug:
        print("Creating numeric preprocessor...")
    
    # Define a pipeline for numeric features with constant imputation and scaling
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Set missing values to zero
        ('scaler', StandardScaler())  # Scale numerical features
    ])
    
    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features)
    ])
    
    if debug:
        print(f"Numeric preprocessing pipeline created for features: {numeric_features}")
    
    return preprocessor

# Function to perform clustering and return the labeled dataset
def perform_clustering(data, defensive_features, n_clusters=3, debug=False):
    """
    Perform K-Means and DBSCAN clustering, add cluster labels to the dataset, and return the updated dataset.
    
    Parameters:
    - data: The preprocessed DataFrame containing defensive features.
    - defensive_features: List of defensive features to use for clustering.
    - n_clusters: Number of clusters for K-Means.
    - debug: If True, display detailed outputs, visualizations, and analytics.
    
    Returns:
    - Updated DataFrame with cluster labels.
    """
    # Preprocess the defensive features
    preprocessor = get_numeric_preprocessor(numeric_features=defensive_features, debug=debug)
    X_preprocessed = preprocessor.fit_transform(data[defensive_features])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['defensive_cluster_kmeans'] = kmeans.fit_predict(X_preprocessed)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    data['defensive_cluster_dbscan'] = dbscan.fit_predict(X_preprocessed)

    # Assign descriptive labels to K-Means clusters based on analysis
    cluster_labels = {0: 'Quick Reactors', 1: 'Moderate Defenders', 2: 'Late Reactors'}
    data['defensive_cluster_label'] = data['defensive_cluster_kmeans'].map(cluster_labels)

    # Calculate detailed cluster statistics for K-Means clusters
    cluster_summary = data.groupby('defensive_cluster_kmeans')[defensive_features].agg(['mean', 'median', 'std', 'min', 'max'])

    if debug:
        # Print cluster statistics
        print("Detailed K-Means Cluster Summary:\n", cluster_summary)

        # Visualize K-Means Clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_preprocessed[:, 0], y=X_preprocessed[:, 1], hue=data['defensive_cluster_kmeans'], palette='viridis')
        plt.title('K-Means Clusters based on Defensive Stats')
        plt.show()

        # Visualize DBSCAN Clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_preprocessed[:, 0], y=X_preprocessed[:, 1], hue=data['defensive_cluster_dbscan'], palette='coolwarm')
        plt.title('DBSCAN Clusters based on Defensive Stats')
        plt.show()

        # Plot distributions of key features across clusters
        for feature in defensive_features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='defensive_cluster_kmeans', y=feature, data=data, palette='Set2')
            plt.title(f'Distribution of {feature} Across K-Means Clusters')
            plt.show()

        # Pairplot to visualize pairwise relationships between features for each cluster
        sns.pairplot(data, hue='defensive_cluster_kmeans', vars=defensive_features, palette='viridis')
        plt.suptitle('Pairplot of Defensive Features by K-Means Clusters', y=1.02)
        plt.show()

        # Calculate inter-cluster distances for K-Means clusters
        cluster_centers = kmeans.cluster_centers_
        distances = cdist(X_preprocessed, cluster_centers, 'euclidean')
        data['distance_to_cluster_center'] = distances.min(axis=1)
        print("Average distance to cluster centers for each cluster:\n", data.groupby('defensive_cluster_kmeans')['distance_to_cluster_center'].mean())

        # Visualize the distribution of distances to cluster centers
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='defensive_cluster_kmeans', y='distance_to_cluster_center', data=data, palette='Set3')
        plt.title('Distance to Cluster Centers by K-Means Cluster')
        plt.show()

        # Visualize average values of each feature for each cluster using bar plots
        cluster_mean_values = data.groupby('defensive_cluster_kmeans')[defensive_features].mean().reset_index()
        cluster_mean_values_melted = cluster_mean_values.melt(id_vars='defensive_cluster_kmeans', var_name='Feature', value_name='Mean Value')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='defensive_cluster_kmeans', y='Mean Value', hue='Feature', data=cluster_mean_values_melted, palette='tab10')
        plt.title('Mean Value of Defensive Features for Each K-Means Cluster')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    return data

# Main function to run clustering analysis and return labeled dataset
def feature_engineering_with_cluster_analysis(df, debug=False):
    """
    Main function to load data, perform clustering, and return the labeled dataset.
    
    Parameters:
    - df: Input DataFrame with necessary features.
    - debug: If True, enable debug mode and output additional analytics and visualizations.
    
    Returns:
    - Labeled DataFrame with cluster labels.
    """

    # Select only the defensive-related stats for clustering
    defensive_features = ['reaction_speed', 'distance_covered', 'catch_probability']

    # Perform clustering and get the labeled data
    df = perform_clustering(df, defensive_features, n_clusters=3, debug=debug)

    return df


def display_cluster_analysis(data):
    st.subheader("Cluster Analysis Visualizations")
    # Scatter plot for K-Means clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['reaction_speed'], y=data['distance_covered'], hue=data['defensive_cluster_kmeans'], palette='viridis', ax=ax)
    ax.set_title("K-Means Clusters based on Defensive Stats")
    st.pyplot(fig)

    # Boxplot for each defensive feature by cluster
    for feature in ['reaction_speed', 'distance_covered', 'catch_probability']:
        fig, ax = plt.subplots()
        sns.boxplot(x='defensive_cluster_kmeans', y=feature, data=data, palette='Set2', ax=ax)
        ax.set_title(f'Distribution of {feature} Across K-Means Clusters')
        st.pyplot(fig)
        

# Example usage
if __name__ == "__main__":
    # Load preprocessed data (adjust the file path as needed)
    file_path = "../../data/Seattle Mariners 2025 Analytics Internship/data-train-preprocessed.csv"
    
    # Run clustering with debug mode enabled to visualize outputs
    labeled_df = feature_engineering_with_cluster_analysis(pd.read_csv(file_path), debug=True)
    print("Labeled DataFrame Head:\n", labeled_df.head())
    print("Labeled DataFrame columns:\n", labeled_df.columns)
    
    display_cluster_analysis(pd.read_csv(file_path))

# Analyze the clusters to find meaningful labels
# For example:
# Cluster 0: High Reaction Speed and Distance Covered = "Quick Reactors"
# Cluster 1: Moderate Reaction Speed and Low Distance = "Moderate Defenders"
# Cluster 2: Low Reaction Speed and High Distance = "Late Reactors"

# clusters cause data leakage, great for post prediction analsis on these players
