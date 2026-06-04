# Titanic Survival Prediction

This is a complete Data Science internship project that predicts Titanic survival using a real-world dataset.
It demonstrates the full end-to-end workflow:
- Data collection and dataset understanding
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Data visualization with Matplotlib and Seaborn
- Predictive modeling with Scikit-Learn
- Model evaluation and reporting

## Project Structure

```
TitanicSurvivalProject/
├── data/
│   └── Titanic-Dataset.csv        # Titanic dataset CSV generated or loaded from seaborn
├── output/
│   ├── accuracy_report.txt        # Saved model report and metrics
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── gender_distribution.png
│   ├── age_distribution.png
│   ├── survival_by_gender_pie.png
│   └── survival_count.png
├── titanic_survival.py            # Main analysis script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

## Dataset

- The project uses the Titanic dataset.
- If `data/Titanic-Dataset.csv` is missing, the script loads the Titanic dataset from `seaborn` and saves it locally.
- This ensures the project runs without a manual dataset download.

## Installation

1. Create a virtual environment from the project folder:
```bash
python -m venv .venv
```

2. Activate the environment:
- PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
- CMD:
```cmd
.venv\Scripts\activate.bat
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the Project

From `TitanicSurvivalProject`:

```bash
python titanic_survival.py
```

The script will:
- Load the Titanic dataset
- Display the first 5 rows, shape, columns, and data types
- Clean missing values and remove duplicates
- Perform EDA and print insights
- Create charts in `output/`
- Train a Logistic Regression model
- Save evaluation metrics and plots

## Sample Outputs

- `output/survival_count.png`
- `output/gender_distribution.png`
- `output/age_distribution.png`
- `output/class_distribution.png`
- `output/survival_by_gender_pie.png`
- `output/correlation_heatmap.png`
- `output/confusion_matrix.png`
- `output/model_report.txt`

## Model Results

- The project trains a Logistic Regression model on the Titanic dataset.
- It prints accuracy and classification metrics to the console.
- It also saves a confusion matrix and model report to `output/`.

## GitHub Push Commands

```bash
git add TitanicSurvivalProject/
git commit -m "Add Titanic Survival Prediction internship project"
git push origin main
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Notes

- The code is fully commented and organized for readability.
- The dataset is prepared for analysis and modeling.
- Visualizations include titles, labels, and legends.
