# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
dataset_path = "D:\WebApplication\data.csv"  # Provide the correct path
df = pd.read_csv(dataset_path)

# Display basic dataset information
print("Dataset Overview:")
print(df.head(), "\n")
print("Dataset Structure:")
print(df.info(), "\n")
print("Missing Values in Each Column:")
print(df.isnull().sum(), "\n")

# Data Preprocessing
df.columns = df.columns.str.strip()  # Trim spaces from column names

# Encode categorical variables using LabelEncoder
categorical_features = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test', 'Class/ASD Traits']
encoder_dict = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    encoder_dict[feature] = encoder  # Store encoders for future use

# Define features (X) and target variable (y)
X = df.drop(columns=['Class/ASD Traits', 'Case_No'])
y = df['Class/ASD Traits']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Model Prediction
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No ASD", "ASD"], yticklabels=["No ASD", "ASD"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Display dataset overview
print("Dataset Overview:")
print(df.head(), "\n")
print("Dataset Structure:")
print(df.info(), "\n")
print("Missing Values in Each Column:")
print(df.isnull().sum(), "\n")

# Data Preprocessing
df.columns = df.columns.str.strip()  # Trim spaces from column names

# Encode categorical variables using LabelEncoder
categorical_features = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test', 'Class/ASD Traits']
encoder_dict = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    encoder_dict[feature] = encoder  # Store encoders for future use

# Define features (X) and target variable (y)
X = df.drop(columns=['Class/ASD Traits', 'Case_No'])
y = df['Class/ASD Traits']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the Support Vector Machine (SVM) Classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Radial Basis Function (RBF) kernel
svm_model.fit(X_train, y_train)

# Model Prediction
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No ASD", "ASD"], yticklabels=["No ASD", "ASD"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Display dataset overview
print("Dataset Overview:")
print(df.head(), "\n")
print("Dataset Structure:")
print(df.info(), "\n")
print("Missing Values in Each Column:")
print(df.isnull().sum(), "\n")

# Data Preprocessing
df.columns = df.columns.str.strip()  # Trim spaces from column names

# Encode categorical variables using LabelEncoder
categorical_features = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test', 'Class/ASD Traits']
encoder_dict = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    df[feature] = encoder.fit_transform(df[feature])
    encoder_dict[feature] = encoder  # Store encoders for future use

# Define features (X) and target variable (y)
X = df.drop(columns=['Class/ASD Traits', 'Case_No'])
y = df['Class/ASD Traits']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Model Prediction
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No ASD", "ASD"], yticklabels=["No ASD", "ASD"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
