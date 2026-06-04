# Titanic Survival Prediction

A complete internship-ready data science project for predicting Titanic survival using Python.

## Project Overview

This project demonstrates the end-to-end data science workflow:
- Data collection and dataset understanding
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Data visualization
- Predictive modeling with Logistic Regression
- Model evaluation and reporting

## Project Structure

```
TitanicSurvivalProject/
├── Titanic-Dataset.csv       # Titanic dataset (place your CSV here)
├── titanic_survival.py       # Main Python analysis script
├── requirements.txt         # Required Python packages
├── README.md                # Documentation
├── .gitignore               # Git ignore rules
├── survival_count.png       # Generated visualizations
├── gender_distribution.png
├── age_distribution.png
├── class_distribution.png
├── survival_by_gender_pie.png
├── correlation_heatmap.png
└── confusion_matrix.png
```

## Dataset Requirements

- Download the Titanic dataset CSV file and place it in the `TitanicSurvivalProject` folder.
- Name the file: `Titanic-Dataset.csv`

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```
2. Activate the virtual environment:
- Windows PowerShell:
```powershell
venv\Scripts\Activate.ps1
```
- Windows CMD:
```cmd
venv\Scripts\activate.bat
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python titanic_survival.py
```

## What the Script Does

1. Loads the Titanic dataset using Pandas.
2. Displays the first 5 rows, shape, and data types.
3. Explains each feature in the dataset.
4. Handles missing values and removes duplicate rows.
5. Converts categorical values into numeric values.
6. Calculates summary statistics and survival trends.
7. Creates visualizations and saves them as PNG files.
8. Trains a Logistic Regression model and evaluates performance.

## Results and Conclusions

- The dataset is processed and cleaned for modeling.
- Survival rates are analyzed by age, gender, and passenger class.
- Logistic Regression is used to predict survival.
- Model accuracy, confusion matrix, and classification report are displayed.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## GitHub Ready

- Professional project structure.
- Clear code comments and modular sections.
- `requirements.txt` included.
- `.gitignore` included.
- Ready to upload to GitHub for internship submission.
