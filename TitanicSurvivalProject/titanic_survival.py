import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')


def get_project_paths() -> tuple[str, str]:
    """Return absolute paths for the data and output folders."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return data_dir, output_dir


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Titanic dataset column names."""
    rename_map = {
        'survived': 'Survived',
        'pclass': 'Pclass',
        'sex': 'Sex',
        'age': 'Age',
        'sibsp': 'SibSp',
        'parch': 'Parch',
        'fare': 'Fare',
        'embarked': 'Embarked',
        'passengerid': 'PassengerId'
    }
    lower_cols = {col.lower(): col for col in df.columns}
    df = df.rename(columns={
        actual: rename_map[actual.lower()]
        for actual in df.columns
        if actual.lower() in rename_map
    })
    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """Load Titanic data from CSV or fallback to the seaborn Titanic dataset."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print('\n✅ Dataset loaded from CSV file.')
    else:
        print('\n⚠️  Dataset CSV not found. Loading seaborn Titanic dataset as fallback.')
        df = sns.load_dataset('titanic')
        df = standardize_columns(df)
        df['PassengerId'] = range(1, len(df) + 1)
        df['Name'] = np.nan
        df['Ticket'] = np.nan
        df['Cabin'] = np.nan
        desired_order = [
            'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
        ]
        df = df[[col for col in desired_order if col in df.columns]]
        df.to_csv(csv_path, index=False)
        print(f'✅ Saved fallback dataset to: {csv_path}')

    df = standardize_columns(df)
    return df


def dataset_overview(df: pd.DataFrame) -> None:
    """Display dataset preview, dimensions, types, and meaning."""
    print('\n=== TASK 1: DATA COLLECTION & DATASET UNDERSTANDING ===')
    print('\nFirst 5 rows of the dataset:')
    print(df.head())

    print('\nDataset shape:', df.shape)
    print('\nColumns and data types:')
    print(df.dtypes)

    print('\nDataset explanation:')
    print('This dataset represents Titanic passengers, their demographics, travel class, and survival status.')
    print('It is commonly used to demonstrate classification, preprocessing, and model evaluation.')

    feature_info = {
        'PassengerId': 'Unique passenger identifier',
        'Survived': 'Survived (0 = No, 1 = Yes)',
        'Pclass': 'Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)',
        'Sex': 'Passenger gender',
        'Age': 'Passenger age in years',
        'SibSp': 'Number of siblings/spouses aboard',
        'Parch': 'Number of parents/children aboard',
        'Fare': 'Ticket fare',
        'Embarked': 'Port of embarkation (S, C, Q)'
    }
    print('\nFeature descriptions:')
    for feature, desc in feature_info.items():
        if feature in df.columns:
            print(f'  - {feature}: {desc}')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, drop duplicates, and encode categorical columns."""
    print('\n=== TASK 2: DATA CLEANING & PREPROCESSING ===')
    df = df.copy()

    duplicates = df.duplicated().sum()
    print(f'\nDuplicate rows found: {duplicates}')
    df = df.drop_duplicates()

    missing_before = df.isnull().sum()
    print('\nMissing values before cleaning:')
    print(missing_before)

    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].astype(str).str.upper().map({'S': 0, 'C': 1, 'Q': 2})

    drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df.columns]
    df = df.drop(columns=drop_cols)

    missing_after = df.isnull().sum()
    print('\nMissing values after cleaning:')
    print(missing_after)

    print('\nData cleaning completed. The dataset is ready for analysis.')
    return df


def eda(df: pd.DataFrame) -> None:
    """Print descriptive statistics and identify key trends."""
    print('\n=== TASK 3: EXPLORATORY DATA ANALYSIS (EDA) ===')
    print('\nDescriptive statistics:')
    print(df.describe(include='all'))

    if 'Survived' in df.columns:
        survival_rate = df['Survived'].mean() * 100
        print(f'\nOverall survival rate: {survival_rate:.2f}%')

    if {'Sex', 'Survived'}.issubset(df.columns):
        print('\nSurvival rate by gender:')
        print(df.groupby('Sex')['Survived'].mean() * 100)

    if {'Pclass', 'Survived'}.issubset(df.columns):
        print('\nSurvival rate by passenger class:')
        print(df.groupby('Pclass')['Survived'].mean() * 100)

    if {'Age', 'Survived'}.issubset(df.columns):
        age_groups = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100])
        print('\nSurvival rate by age group:')
        print(df.groupby(age_groups)['Survived'].mean() * 100)

    print('\nKey findings:')
    print('  - Female passengers survive at a higher rate than male passengers.')
    print('  - Passengers in first class have better survival odds.')
    print('  - Younger passengers show improved survival percentages.')


def plot_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """Generate charts using Matplotlib and Seaborn and save them to output."""
    print('\n=== TASK 4: DATA VISUALIZATION ===')

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Survived')
    plt.title('Survival Count')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'survival_count.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Sex')
    plt.title('Passenger Gender Distribution')
    plt.xlabel('Sex (0 = male, 1 = female)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['Age'].dropna(), bins=20, kde=False)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), bbox_inches='tight')
    plt.close()

    if 'Pclass' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Pclass')
        plt.title('Passenger Class Distribution')
        plt.xlabel('Passenger Class')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), bbox_inches='tight')
        plt.close()

    if {'Sex', 'Survived'}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        survived_by_gender = df.groupby('Sex')['Survived'].mean() * 100
        survived_by_gender.index = survived_by_gender.index.map({0: 'Male', 1: 'Female'})
        survived_by_gender.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.title('Survival Rate by Gender')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'survival_by_gender_pie.png'), bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

    print(f'\nSaved visualizations to {output_dir}')


def build_model(df: pd.DataFrame, output_dir: str) -> None:
    """Train a logistic regression model and save performance metrics."""
    print('\n=== TASK 5: PREDICTIVE MODEL ===')

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'\nAccuracy on test set: {accuracy * 100:.2f}%')
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(report)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    with open(os.path.join(output_dir, 'model_report.txt'), 'w', encoding='utf-8') as file:
        file.write(f'Accuracy: {accuracy:.4f}\n\n')
        file.write('Confusion Matrix:\n')
        file.write(np.array2string(cm) + '\n\n')
        file.write('Classification Report:\n')
        file.write(report)

    print(f'\nSaved model report to {os.path.join(output_dir, 'model_report.txt')}')


def main() -> None:
    data_dir, output_dir = get_project_paths()
    dataset_path = os.path.join(data_dir, 'Titanic-Dataset.csv')

    df = load_data(dataset_path)
    dataset_overview(df)
    df_cleaned = clean_data(df)
    eda(df_cleaned)
    plot_visualizations(df_cleaned, output_dir)
    build_model(df_cleaned, output_dir)

    print('\n✅ Titanic Survival Prediction project completed successfully.')


if __name__ == '__main__':
    main()
