import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')


def load_data(path: str) -> pd.DataFrame:
    """Load the Titanic dataset from a CSV file."""
    try:
        df = pd.read_csv(path)
        print("\n✅ Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place Titanic-Dataset.csv in the project folder."
        )


def dataset_overview(df: pd.DataFrame) -> None:
    """Display dataset preview, shape, columns, types, and feature descriptions."""
    print("\n=== DATASET OVERVIEW ===")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nColumn names and data types:")
    print(df.dtypes)

    print("\nFeature descriptions:")
    feature_info = {
        'PassengerId': 'Passenger identifier',
        'Survived': 'Survival (0 = No, 1 = Yes)',
        'Pclass': 'Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)',
        'Name': 'Passenger name',
        'Sex': 'Gender of passenger',
        'Age': 'Age in years',
        'SibSp': 'Number of siblings/spouses aboard',
        'Parch': 'Number of parents/children aboard',
        'Ticket': 'Ticket number',
        'Fare': 'Passenger fare',
        'Cabin': 'Cabin number',
        'Embarked': 'Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)'
    }
    for feature, explanation in feature_info.items():
        if feature in df.columns:
            print(f"  - {feature}: {explanation}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset and convert categorical data to numeric values."""
    df = df.copy()
    print("\n=== DATA CLEANING & PREPROCESSING ===")

    # Remove duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicates found: {duplicates}")
    df = df.drop_duplicates()

    # Handle missing values
    missing_before = df.isnull().sum()
    print("\nMissing values before cleaning:")
    print(missing_before)

    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Drop columns that are not useful for modeling
    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    for col in drop_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Convert categorical columns
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    missing_after = df.isnull().sum()
    print("\nMissing values after cleaning:")
    print(missing_after)

    return df


def eda(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis and print insights."""
    print("\n=== EXPLORATORY DATA ANALYSIS (EDA) ===")

    print("\nSummary statistics:")
    print(df.describe(include='all'))

    if 'Survived' in df.columns:
        survival_rate = df['Survived'].mean() * 100
        print(f"\nOverall survival rate: {survival_rate:.2f}%")

    if {'Age', 'Survived'}.issubset(df.columns):
        age_survival = df.groupby(pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100]))['Survived'].mean()
        print("\nSurvival rate by age group:")
        print(age_survival)

    if {'Sex', 'Survived'}.issubset(df.columns):
        gender_survival = df.groupby('Sex')['Survived'].mean() * 100
        print("\nSurvival rate by gender:")
        print(gender_survival)

    if {'Pclass', 'Survived'}.issubset(df.columns):
        class_survival = df.groupby('Pclass')['Survived'].mean() * 100
        print("\nSurvival rate by passenger class:")
        print(class_survival)

    print("\nInsights:")
    print("  - Women tend to survive at a higher rate than men.")
    print("  - First class passengers show a higher survival percentage.")
    print("  - Children and younger adults have better survival rates in this dataset.")


def plot_visualizations(df: pd.DataFrame) -> None:
    """Create and save visualizations for the Titanic dataset."""
    print("\n=== DATA VISUALIZATION ===")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Survived')
    plt.title('Survival Count')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.savefig('TitanicSurvivalProject/survival_count.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Sex')
    plt.title('Passenger Gender Distribution')
    plt.xlabel('Sex (0=male, 1=female)')
    plt.ylabel('Count')
    plt.savefig('TitanicSurvivalProject/gender_distribution.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['Age'].dropna(), bins=20, kde=False)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('TitanicSurvivalProject/age_distribution.png', bbox_inches='tight')
    plt.close()

    if 'Pclass' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Pclass')
        plt.title('Passenger Class Distribution')
        plt.xlabel('Passenger Class')
        plt.ylabel('Count')
        plt.savefig('TitanicSurvivalProject/class_distribution.png', bbox_inches='tight')
        plt.close()

    if {'Sex', 'Survived'}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        survived_by_gender = df.groupby('Sex')['Survived'].mean() * 100
        survived_by_gender.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.title('Survival Rate by Gender')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig('TitanicSurvivalProject/survival_by_gender_pie.png', bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('TitanicSurvivalProject/correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    print("\nSaved visualization files in the TitanicSurvivalProject folder.")


def build_model(df: pd.DataFrame) -> None:
    """Build and evaluate a logistic regression model for survival prediction."""
    print("\n=== PREDICTIVE MODEL ===")

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

    print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('TitanicSurvivalProject/confusion_matrix.png', bbox_inches='tight')
    plt.close()


def main() -> None:
    dataset_path = 'TitanicSurvivalProject/Titanic-Dataset.csv'
    df = load_data(dataset_path)
    dataset_overview(df)
    df_cleaned = clean_data(df)
    eda(df_cleaned)
    plot_visualizations(df_cleaned)
    build_model(df_cleaned)

    print("\n✅ Titanic Survival Prediction project completed successfully.")


if __name__ == '__main__':
    main()
