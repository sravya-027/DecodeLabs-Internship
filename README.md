# Data Science Internship Project

A comprehensive data analysis and machine learning project designed to demonstrate proficiency in data science fundamentals, exploratory data analysis, and predictive modeling.

## 🎯 Project Overview

This project covers the complete data science workflow from data loading and cleaning to building and evaluating machine learning models. It's structured to showcase essential skills required for a data science internship position.

## 📋 Project Tasks

### Task 1: Dataset Understanding
- Load and explore the dataset
- Display dataset shape, columns, and data types
- Generate statistical summaries
- Identify missing values and data quality issues

### Task 2: Data Cleaning & Preprocessing
- Handle missing values:
  - Numeric columns: filled with median
  - Categorical columns: filled with mode
- Remove duplicate records
- Validate data quality post-cleaning
- Save cleaned dataset for further analysis

### Task 3: Exploratory Data Analysis (EDA)
- Statistical analysis and descriptive statistics
- Correlation analysis between features
- Identify strong correlations (|r| > 0.5)
- Outlier detection using Interquartile Range (IQR) method
- Generate visualizations for data insights

### Task 4: Machine Learning Models
Two regression models trained and compared:
- **Linear Regression**: Fast baseline model with feature scaling
- **Random Forest**: Ensemble method for better generalization

Models trained to predict Salary based on:
- Age
- Years of Experience
- Department
- Performance Score

### Task 5: Model Evaluation & Insights
Performance metrics calculated:
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MAE**: Mean Absolute Error in dollars
- **RMSE**: Root Mean Squared Error (penalizes larger errors)
- **Feature Importance**: Identifies most influential features

## 📊 Visualizations Generated

1. **correlation_heatmap.png** - Feature correlation matrix
2. **distributions.png** - Histograms of numerical features
3. **boxplots.png** - Outlier detection visualization
4. **model_predictions.png** - Actual vs Predicted salary comparison

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 Running the Project

### Option 1: Run from Terminal
```bash
cd path/to/pandasApp
python decode1.py
```

### Option 2: Run from VS Code
1. Open `decode1.py` in VS Code
2. Click the "Run" button or press `Ctrl+F5`
3. View output in the terminal or Python Debug Console

## 📁 Project Files

```
pandasApp/
├── decode1.py              # Main analysis script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── dataset.csv            # Sample/input dataset
├── cleaned_dataset.csv    # Output: cleaned data
├── correlation_heatmap.png
├── distributions.png
├── boxplots.png
└── model_predictions.png
```

## 📈 Expected Output

The script generates:
- Console output with detailed analysis results
- 4 visualization PNG files
- `cleaned_dataset.csv` with processed data

## 🔍 Key Features

✅ **Robust Error Handling**: Auto-generates sample data if dataset not found
✅ **Automatic Data Cleaning**: Handles missing values intelligently
✅ **Multiple ML Models**: Compares Linear Regression vs Random Forest
✅ **Comprehensive Metrics**: R², MAE, RMSE for thorough evaluation
✅ **Feature Importance**: Identifies key predictive features
✅ **High-Quality Visualizations**: Publication-ready chart exports

## 💡 Methodologies Used

- **Missing Value Imputation**: Median (numeric), Mode (categorical)
- **Outlier Detection**: IQR (Interquartile Range) method
- **Feature Scaling**: StandardScaler for optimal model performance
- **Train-Test Split**: 80-20 ratio for unbiased evaluation
- **Ensemble Methods**: Random Forest for improved predictions

## 📊 Model Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| R² Score | Explains variance in data | 0-1 (higher better) |
| MAE | Average prediction error in dollars | Any (lower better) |
| RMSE | Penalizes large errors more | Any (lower better) |
| Feature Importance | Which features matter most | 0-1 |

## 🎓 Learning Outcomes

This project demonstrates:
- ✓ Data loading and exploration
- ✓ Data cleaning and preprocessing
- ✓ Exploratory data analysis techniques
- ✓ Statistical analysis and correlation
- ✓ Machine learning model development
- ✓ Model evaluation and comparison
- ✓ Data visualization best practices
- ✓ Professional Python coding standards

## 🔧 Customization

To use with your own dataset:
1. Replace `dataset.csv` in the project folder
2. Ensure your CSV has similar structure (numeric and categorical columns)
3. The script auto-generates sample data if file not found

## 📝 Sample Output

```
============================================================
DATA SCIENCE INTERNSHIP PROJECT - COMPREHENSIVE ANALYSIS
============================================================

✓ Dataset loaded successfully!

------------------------------------------------------------
TASK 1: DATASET UNDERSTANDING
------------------------------------------------------------

📊 First 5 Records:
   Age  Salary  Years_Experience Department  Performance_Score
0   45   95234                15       Sales                  8
1   32   65432                 8         IT                   7
...

📈 Dataset Shape: 200 rows × 5 columns

[... detailed analysis output ...]

📈 LINEAR REGRESSION MODEL PERFORMANCE:
  Training R² Score: 0.8234
  Testing R² Score: 0.7945
  Mean Absolute Error (MAE): $8234.56
  Root Mean Squared Error (RMSE): $10567.89

🌲 RANDOM FOREST MODEL PERFORMANCE:
  Training R² Score: 0.9156
  Testing R² Score: 0.8567
  Mean Absolute Error (MAE): $6789.23
  Root Mean Squared Error (RMSE): $8956.12

============================================================
✨ PROJECT COMPLETED SUCCESSFULLY! ✨
============================================================
```

## 🐛 Troubleshooting

**Issue**: ModuleNotFoundError
- **Solution**: Install missing packages: `pip install -r requirements.txt`

**Issue**: FileNotFoundError for dataset.csv
- **Solution**: The script auto-creates sample data. No action needed!

**Issue**: Visualization files not created
- **Solution**: Ensure matplotlib backend is configured. Check write permissions.

## 📚 Technologies Used

- **Python 3.7+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualization
- **Scikit-learn**: Machine learning algorithms

## 🎖️ Best Practices Demonstrated

✅ Clear code comments and documentation
✅ Error handling and graceful fallbacks
✅ Modular section organization
✅ Meaningful variable names
✅ Professional output formatting
✅ Version control ready (git-compatible)

## 📞 Support

For questions or improvements:
1. Review the inline code comments
2. Check the console output messages
3. Verify all dependencies are installed
4. Ensure sufficient disk space for visualizations

## 📄 License

This project is created for educational and internship purposes.

---

**Ready to impress your internship interviewers!** 🚀

Last Updated: June 2, 2026
