# 📨 SMS Spam Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=flat-square&logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.6%2B-green?style=flat-square&logo=python&logoColor=white)
![tkinter](https://img.shields.io/badge/tkinter-GUI-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensource&logoColor=white)

A machine learning-based SMS spam classifier that uses Support Vector Machine (SVM) with TF-IDF vectorization to detect spam messages. The application includes a user-friendly GUI built with tkinter for easy interaction.

## 🌐 Live Demo

🚀 **Try the live application**: [Spam Detector App](https://v0-sms-spam.vercel.app/)

This application is deployed on Vercel for fast, reliable performance with global CDN distribution. Vercel provides seamless integration with Git repositories and automatic deployments on every push to the main branch.

## 🚀 Features

- **Machine Learning Model**: SVM classifier with hyperparameter tuning using GridSearchCV
- **Text Preprocessing**: Advanced text cleaning with stemming and stop word removal
- **TF-IDF Vectorization**: Efficient text feature extraction
- **GUI Interface**: Clean, dark-themed interface for real-time spam detection
- **Model Persistence**: Save and load trained models using pickle
- **High Accuracy**: Optimized model performance through parameter tuning

## 📋 Requirements

### 🐍 Python Dependencies
- Python 3.8+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+
- nltk 3.6+
- opencv-python 4.5+
- tkinter (usually included with Python)

### 📊 Data Requirements
- CSV file named `spam.csv` with columns:
  - `Label`: Message classification (spam/ham)
  - `EmailText`: Message content

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd spam-detector
   ```

2. **Install required packages**:
   ```bash
   pip install numpy pandas scikit-learn nltk opencv-python
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Prepare your dataset**:
   - Place your `spam.csv` file in the project directory
   - Ensure it has the required columns: `Label` and `EmailText`

## 🎯 Usage

### Training the Model

Run the script to train the model:
```bash
python spam-detect.py
```

The script will:
1. Load and preprocess the dataset
2. Apply text cleaning and stemming
3. Split data into training and testing sets
4. Train an SVM model with hyperparameter tuning
5. Save the trained model as `finalized_model.sav`

### Using the GUI (Currently Commented)

The GUI code is included but commented out. To enable it:
1. Uncomment the GUI section at the bottom of the script
2. Run the script
3. Enter a message in the text field
4. Click "Check" to classify the message

## 🔧 Model Architecture

### Text Preprocessing Pipeline
1. **Tokenization**: Split text into individual words
2. **Normalization**: Convert to lowercase
3. **Alphanumeric Filtering**: Keep only alphanumeric characters
4. **Stop Word Removal**: Remove common English stop words
5. **Stemming**: Apply Porter Stemming algorithm

### Machine Learning Pipeline
1. **Feature Extraction**: TF-IDF Vectorization
2. **Model**: Support Vector Machine (SVM)
3. **Hyperparameter Tuning**: GridSearchCV with parameters:
   - Kernel: ['linear', 'rbf']
   - Gamma: [1e-3, 1e-4]
   - C: [1, 10, 100, 1000]

## 📊 Model Performance

The model outputs accuracy metrics after training and testing. Performance depends on:
- Quality and size of training data
- Hyperparameter optimization results
- Text preprocessing effectiveness

## 🎨 GUI Features

The graphical interface includes:
- **Dark Theme**: Modern, eye-friendly design
- **Real-time Detection**: Instant spam classification
- **Visual Feedback**: Color-coded results (red for spam, green for legitimate)
- **Input Validation**: Handles empty input gracefully
- **Clear Functionality**: Easy input reset

## 📁 Project Structure

```
spam-detector/
├── spam-detect.py          # Main application file
├── spam.csv               # Training dataset
├── finalized_model.sav    # Saved trained model
└── README.md             # This file
```

## 🔍 Key Functions

- `retrieve_importantFeatures()`: Tokenizes and filters alphanumeric characters
- `remove_stopWords()`: Removes stop words and punctuation
- `potter_stem()`: Applies Porter stemming algorithm
- `check_spam()`: GUI function for real-time classification
- `clear_input()`: GUI function to reset input field

## 🚦 Getting Started

1. Ensure you have all dependencies installed
2. Place your training data (`spam.csv`) in the project directory
3. Run the script to train the model
4. Uncomment the GUI section to use the interactive interface
5. Test with sample messages to verify functionality

## 📈 Model Training Process

The training process includes:
1. **Data Loading**: Read CSV with pandas
2. **Label Encoding**: Convert categorical labels to numerical
3. **Duplicate Removal**: Clean dataset integrity
4. **Text Preprocessing**: Apply cleaning pipeline
5. **Train/Test Split**: 75/25 split with random state 42
6. **Model Training**: SVM with GridSearchCV optimization
7. **Model Evaluation**: Accuracy assessment
8. **Model Persistence**: Save using pickle

## ⚠️ Important Notes

- The GUI section is currently commented out in the provided code
- Ensure NLTK data is downloaded before running
- Model accuracy depends on the quality of your training dataset
- The saved model file (`finalized_model.sav`) is required for GUI functionality
