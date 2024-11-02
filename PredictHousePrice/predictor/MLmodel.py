import joblib
import os

# Load model on app startup
LINEAR_REGRESSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linearRegressionModel.joblib')
linearRegressionModel = joblib.load(LINEAR_REGRESSION_MODEL_PATH)

DECISION_TREE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'decisionTreeModel.joblib')
decisionTreeModel = joblib.load(DECISION_TREE_MODEL_PATH)
