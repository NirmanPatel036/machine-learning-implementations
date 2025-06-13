# Linear Regression from Scratch

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![JavaScript](https://img.shields.io/badge/javascript-ES6+-yellow.svg)
![Chart.js](https://img.shields.io/badge/Chart.js-v3.9+-orange.svg)
![HTML5](https://img.shields.io/badge/HTML5-supported-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive implementation of linear regression using gradient descent, featuring both a Python implementation and an interactive web-based visualization tool.

## Overview

This project demonstrates linear regression concepts through two main components:
- **Python Implementation**: A from-scratch implementation of linear regression using gradient descent
- **Interactive Web Interface**: A real-time visualization tool for understanding gradient descent and regression line fitting

The project uses salary prediction based on years of experience as the primary use case.

## Features

### Python Implementation (`linear.py`)
- **Custom Gradient Descent**: Implementation from scratch without using scikit-learn
- **Data Preprocessing**: Handles CSV data with comma-separated salary values
- **Loss Function**: Mean Squared Error (MSE) implementation
- **Training Visualization**: Scatter plots and regression line visualization
- **Progress Monitoring**: Real-time loss tracking during training

### Interactive Web Interface (`index.html`)
- **Dual Modes**: 
  - Animated training visualization
  - Interactive draggable regression line
- **Real-time Statistics**: Loss, R² score, equation display
- **Dynamic Data Generation**: Generate new synthetic datasets
- **Modern UI**: Glassmorphism design with smooth animations
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
linear-regression/
├── linear.py           # Python implementation with gradient descent
├── index.html          # Interactive web visualization
└── salary_data.csv     # Dataset (years of experience vs salary)
```

## Requirements

### Python Implementation
- Python 3.7+
- pandas
- numpy
- matplotlib

### Web Interface
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Chart.js (loaded via CDN)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd linear-regression
```

2. Install Python dependencies:
```bash
pip install pandas numpy matplotlib
```

## Usage

### Python Implementation

Run the gradient descent training:

```bash
python linear.py
```

This will:
- Load and preprocess the salary dataset
- Display initial data visualization
- Train the model using custom gradient descent
- Show training progress every 200 epochs
- Display final equation and loss
- Generate comparison plots

### Interactive Web Interface

Open `index.html` in your web browser to access the interactive tool.

#### Features:
- **📈 Animated Training Mode**: Watch gradient descent find the optimal line
- **🎮 Draggable Line Mode**: Manually adjust the regression line
- **🎲 Generate New Data**: Create synthetic datasets for experimentation
- **Real-time Stats**: Monitor equation, loss, and R² score

## Algorithm Details

### Gradient Descent Implementation

The project implements gradient descent with the following components:

**Cost Function (MSE):**
```
J(m,c) = (1/n) × Σ(yi - (mxi + c))²
```

**Gradient Calculations:**
```
∂J/∂m = -(2/n) × Σ(xi × (yi - (mxi + c)))
∂J/∂c = -(2/n) × Σ(yi - (mxi + c))
```

**Parameter Updates:**
```
m_new = m_old - α × ∂J/∂m
c_new = c_old - α × ∂J/∂c
```

### Hyperparameters

- **Learning Rate (α)**: 0.0001
- **Epochs**: 1000
- **Initialization**: m = 0, c = 0

## Dataset

The salary dataset contains:
- **Features**: Years of Experience (1-10 years)
- **Target**: Annual Salary ($30K-$120K range)
- **Size**: ~30 data points
- **Format**: CSV with comma-separated salary values

### Sample Data Structure:
```csv
YearsExperience,Salary
1.2,"39,344"
2.1,"43,526"
3.0,"56,643"
```

## Mathematical Concepts

### Linear Regression Equation
```
y = mx + c
```
Where:
- `y`: Predicted salary
- `x`: Years of experience
- `m`: Slope (salary increase per year)
- `c`: Y-intercept (base salary)

### Performance Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ(actual - predicted)²
```

**R² Score (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
```

## Web Interface Controls

### Animation Mode
- **▶️ Start Training**: Begin gradient descent animation
- **⏸️ Pause**: Pause the training process
- **🔄 Reset**: Reset parameters to initial values
- **🎲 New Dataset**: Generate fresh synthetic data

### Interactive Mode
- **Click & Drag**: Manually adjust the regression line
- **Real-time Updates**: See equation and statistics change instantly

## Technical Implementation

### Python Features
- **Pandas Integration**: Efficient data manipulation
- **Custom Loss Function**: Educational implementation
- **Progress Tracking**: Epoch-by-epoch monitoring
- **Matplotlib Visualization**: Professional plotting

### Web Features
- **Chart.js Integration**: Interactive plotting library
- **Modern CSS**: Glassmorphism and gradient effects
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Smooth animations and interactions

## Educational Value

This project is ideal for:
- **Understanding Gradient Descent**: Visual representation of optimization
- **Linear Regression Concepts**: Hands-on parameter adjustment
- **Algorithm Implementation**: From-scratch coding experience
- **Interactive Learning**: Real-time experimentation

## Performance

- **Python Training**: ~1000 epochs in seconds
- **Web Animation**: 60 FPS smooth visualization
- **Data Generation**: Instant synthetic dataset creation
- **Interactive Updates**: Real-time parameter adjustment
