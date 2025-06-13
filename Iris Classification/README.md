# Iris Species Classification

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-v2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A machine learning project that classifies Iris flower species using multiple algorithms and provides a web interface for predictions.

## Overview

This project implements a complete machine learning pipeline for classifying Iris flowers into three species:
- Iris-setosa
- Iris-versicolor  
- Iris-virginica

The classification is based on four features: sepal length, sepal width, petal length, and petal width.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Multiple Models**: Implementation of Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers
- **Model Persistence**: Trained models are saved for future use
- **Web Interface**: Flask-based web application for interactive predictions
- **Visualizations**: Correlation heatmaps and scatter plots for data insights

## Project Structure

```
iris-classification/
├── iris.py              # Main analysis and model training script
├── deploy_iris.py       # Flask web application
├── Iris.csv            # Dataset
├── saved_model.sav     # Trained model (generated)
├── database.sqlite     # Database file
├── plots/              # Generated visualization plots
│   ├── sepals.png
│   └── petals.png
└── templates/          # HTML templates for Flask app
    └── index.html
```

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Flask
- joblib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd iris-classification
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn flask joblib
```

## Usage

### Training Models

Run the main analysis script to train models and generate visualizations:

```bash
python iris.py
```

This will:
- Load and preprocess the Iris dataset
- Generate exploratory data analysis plots
- Train three different classification models
- Save the best performing model
- Display accuracy scores for each model

### Web Application

Start the Flask web application:

```bash
python deploy_iris.py
```

Navigate to `http://localhost:5000` in your browser to access the web interface where you can:
- Input flower measurements
- Get real-time species predictions
- View prediction results

## Model Performance

The project implements and compares three algorithms:

- **Logistic Regression**: Linear classification approach
- **K-Nearest Neighbors**: Instance-based learning algorithm  
- **Decision Tree**: Tree-based classification model

Each model's accuracy is displayed during training, allowing for performance comparison.

## Data Preprocessing

- **Missing Values**: Checks and handles null values
- **Feature Encoding**: Label encoding for species classification
- **Data Splitting**: 70% training, 30% testing split
- **Feature Selection**: Uses all four measurement features

## Visualizations

The project generates several plots stored in the `plots/` directory:

- **Feature Histograms**: Distribution of each measurement
- **Sepal Scatter Plot**: Sepal length vs width by species
- **Petal Scatter Plot**: Petal length vs width by species  
- **Correlation Heatmap**: Feature correlation matrix

## API Endpoints

### Web Application Routes

- `GET /`: Home page with prediction form
- `POST /predict`: Accepts flower measurements and returns species prediction

## Input Parameters

For predictions, provide the following measurements (in cm):

- **Sepal Length**: Length of the sepal
- **Sepal Width**: Width of the sepal
- **Petal Length**: Length of the petal
- **Petal Width**: Width of the petal

## Example Usage

```python
# Example prediction input
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

# Expected output: Iris-setosa
```

## Dataset

The project uses the classic Iris dataset containing:
- 150 samples
- 4 features (sepal length/width, petal length/width)
- 3 classes (50 samples each)
- No missing values

## Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **Flask**: Web framework
- **pandas**: Data manipulation
- **matplotlib/seaborn**: Data visualization
- **numpy**: Numerical computing
- **joblib**: Model serialization