import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load and preprocess the dataset
dataset_path = "D:/WebApplication/data.csv"  # Ensure this path is correct
df = pd.read_csv(dataset_path)

# Encode categorical variables
categorical_features = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test', 'Class/ASD Traits']
encoder_dict = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    encoder_dict[feature] = encoder

# Define features (X) and target variable (y)
X = df.drop(columns=['Class/ASD Traits', 'Case_No'])
y = df['Class/ASD Traits']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save the model and encoders
joblib.dump(model, "D:/WebApplication/prediction/autism/ml_model/autism_model.pkl")
joblib.dump(scaler, "D:/WebApplication/prediction/autism/ml_model/scaler.pkl")
joblib.dump(encoder_dict, "D:/WebApplication/prediction/autism/ml_model/encoders.pkl")

# Function to make predictions
def predict_asd(data):
    """
    Predicts whether a person has ASD based on input data.
    """
    # Load saved models
    model = joblib.load("D:/WebApplication/prediction/autism/ml_model/autism_model.pkl")
    scaler = joblib.load("D:/WebApplication/prediction/autism/ml_model/scaler.pkl")
    encoder_dict = joblib.load("D:/WebApplication/prediction/autism/ml_model/encoders.pkl")

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([data])

    # Encode categorical features
    for feature in encoder_dict:
        if feature in input_df:
            input_df[feature] = encoder_dict[feature].transform(input_df[feature])

    # Standardize numerical features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    
    return "ASD" if prediction[0] == 1 else "No ASD"
