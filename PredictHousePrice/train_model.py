import os
import pandas as pd
import joblib
from sklearn.feature_selection import VarianceThreshold
from linearRegression import train_linear_regression
from decisionTree import train_decision_tree


# Load the data
def load_data():
    # url = ("https://drive.google.com/uc?export=download&id=14aM0y_Q8lHAnsixprClRA-0an9ZmCwzu")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__ = train_model.py
    data_path = os.path.join(BASE_DIR, 'PredictHousePrice', 'data', 'train.csv')
    df = pd.read_csv(data_path)
    return df


# Get missing data
def get_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

# Handle missing data
def handle_missing_data(df, missing_data):
    df = df.drop(missing_data[missing_data['Total'] > 1].index, axis=1, errors='ignore')
    df = df.drop(df.loc[df['Electrical'].isnull()].index, errors='ignore')
    return df


# Handle Outliers
def handle_outliers(df):
    from scipy import stats
    z_scores = stats.zscore(df['GrLivArea'])
    outliers = df[(z_scores > 3) | (z_scores < -3)]
    outliers = outliers.sort_values(by='GrLivArea', ascending=False)
    df = df.drop(df[(df['Id'] == 1299) | (df['Id'] == 524)].index, errors='ignore')
    df = df.drop('Id', axis=1)
    return df


# Data processing
def process_data(df):
    # Dimensionality Reduction with PCA
    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']

    # Convert all object-type features into numerical format
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    object_columns = X.select_dtypes(include=['object']).columns
    for column in object_columns:
        X[column] = label_encoder.fit_transform(X[column])

    # Scale the data
    from sklearn.preprocessing import StandardScaler
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, 'predictor/pcaModel.joblib')

    # Data splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Train the model
def training(X_train, X_test, y_train, y_test):
    model_choice = input("Enter 0 (Linear Regression) or 1 (Decision Tree): ")
    if model_choice == '0':
        train_linear_regression(X_train, X_test, y_train, y_test)
    elif model_choice == '1':
        train_decision_tree(X_train, X_test, y_train, y_test)
    else:
        print("Invalid input. Please enter 0 or 1.")

