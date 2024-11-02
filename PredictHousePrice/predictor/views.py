import pandas as pd
import joblib
import torch
from django.shortcuts import render
from django.http import HttpResponse
from train_model import get_missing_data


# Create your views here.
def upload_csv(request):
    if request.method == "POST" and request.FILES.get('csvfile'):
        # Retrieve the uploaded file
        csv_file = request.FILES['csvfile']

        # Read the file into a Pandas DataFrame
        df = pd.read_csv(csv_file)

        # Handle missing data
        missing_data = get_missing_data(df)
        df1 = df.drop(missing_data[missing_data['Total'] > 1].index, axis=1, errors='ignore')
        df1 = df1.dropna()
        df1 = df1.drop('Id', axis=1)

        # Convert all object-type features into numerical format
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        object_columns = df1.select_dtypes(include=['object']).columns
        for column in object_columns:
            df1[column] = label_encoder.fit_transform(df1[column])

        # Scale the data
        from sklearn.preprocessing import StandardScaler
        standard_scaler = StandardScaler()
        df1_scaled = standard_scaler.fit_transform(df1)

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=46)
        df1_pca = pca.fit_transform(df1_scaled)
        
        # Use the trained model (linear regression) to predict
        global df1_final, predict, result
        if request.POST['type'] == "Linear Regression":
            linearRegressionModel = joblib.load('predictor/linearRegressionModel.joblib')
            df1_tensor = linearRegressionModel(torch.tensor(df1_pca, dtype=torch.float32, requires_grad=True))
            df1_final = df1_tensor.detach().numpy()
            predict = pd.DataFrame(df1_final, columns=['SalePrice'])
        else:
            decisionTreeModel = joblib.load('predictor/decisionTreeModel.joblib')
            df1_final = decisionTreeModel.predict(df1_pca)
            predict = pd.DataFrame(df1_final, columns=['SalePrice'])
        
        result = pd.concat([df1, predict], axis=1)

        # Render the table in the response
        return HttpResponse(f"<h2>CSV Data:</h2>{result.to_html()}")


# Default page
def index(request):
    return render(request, 'index.html')
