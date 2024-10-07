# Minor League Outfielder Prediction App

This project uses Streamlit to build an interactive web application for predicting air-out probabilities of batted balls in Minor League Baseball. It incorporates data processing, feature engineering, and model evaluation, while providing an interactive interface for user predictions. The app is designed for the Seattle Mariners 2025 Analytics Internship application and demonstrates various aspects of data science, including machine learning, data visualization, and scouting report generation.

## App URL
[Streamlit App: Outfielder Machine Learning](https://outfielder-machine-learning.streamlit.app)

---

## Project Structure

```plaintext
project_root/
│
├── .devcontainer/
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── devcontainer.env
│   ├── environment.yml
│   ├── requirements.txt
│   ├── .dockerignore
│   ├── install_dependencies.sh
│   ├── install_quarto.sh
│   └── install_requirements.sh
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── mariners_data_streamlit.py  # Main Streamlit app script
│
├── app/
│   └── ...
│
├── tests/
│   └── test1.py
│
├── README.md
│
└── docker-compose.yml
```

---

## Features

1. **Data Preprocessing**: Handles missing values, drops irrelevant columns, and standardizes features.
2. **Feature Engineering**: Calculates physical features such as hit trajectories and adjusted distances.
3. **Model Training**: Trains several models, ultimately selecting CatBoost due to its strong performance with categorical features.
4. **Prediction Interface**: Provides a user-friendly form to input play data and predict the air-out probability.
5. **Scouting Report Generation**: Allows users to generate comprehensive scouting reports based on the predictions.
6. **Visualization**: Visualizes hit trajectories and outfielder performance.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project_root/
```

### 2. Open in VSCode

Make sure you have the DevContainer extension installed in VSCode.

1. Open the project folder in VSCode.
2. Click on the **Remote - Containers** extension and choose `Open Folder in Container`.
3. This will build the Docker environment and install the required dependencies.

### 3. Build and Run the Docker Environment

```bash
# Build the Docker container
docker-compose up --build

# To run the container
docker-compose up
```

### 4. Access the Streamlit App

Once the Docker container is up and running, you can access the Streamlit app at:

```bash
http://localhost:8501
```

---

## Usage

- The **Information Section** provides an overview of the data, feature engineering, and modeling approach.
- The **Prediction Section** allows users to input data and get predictions for air-out probabilities.
- The **Scouting Report Section** generates a detailed report for selected player IDs.
- The **Mariners Improvements Section** offers insights into potential strategic adjustments for the team.

### Commands

- To install dependencies:

```bash
conda env create -f .devcontainer/environment.yml
```

- To run tests:

```bash
pytest tests/
```

---

## Future Improvements

1. Integrate **park factors** and **air density** to improve predictions.
2. Enhance **defensive metrics** with more granular data like Ultimate Zone Rating (UZR) and Defensive Runs Saved (DRS).
3. Add support for **dynamic venue dimensions** for better home run and foul ball catchability predictions.
4. Implement a **LLM RAG bot** to automate the generation of scouting reports and data analysis.

---

This README provides a brief overview of the project, its structure, and instructions on setting up and running the application. For more details on the methodology and results, refer to the project documentation and code comments.