from train_model import load_data
from train_model import get_missing_data
from train_model import handle_missing_data
from train_model import handle_outliers
from train_model import process_data
from train_model import training


df = load_data()
missing_data = get_missing_data(df)
df = handle_missing_data(df, missing_data)
df = handle_outliers(df)
X_train, X_test, y_train, y_test = process_data(df)
training(X_train, X_test, y_train, y_test)
