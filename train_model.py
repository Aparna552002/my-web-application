import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the dataset
dataset_path = "D:/WebApplication/data.csv"
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

print("Class Distribution Before SMOTE:")
print(df["Class/ASD Traits"].value_counts())

# Define features (X) and target variable (y)
X = df.drop(columns=['Class/ASD Traits', 'Case_No'])
y = df['Class/ASD Traits']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Class Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train & Evaluate Multiple Models
models = {
    "Logistic Regression": LogisticRegression(solver="saga", max_iter=1000),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
}

accuracy_scores = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_scores[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No ASD", "ASD"], yticklabels=["No ASD", "ASD"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Accuracy Comparison Plot
plt.figure(figsize=(8, 5), dpi=150)
ax = sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="deep")

# Set labels and title
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Algorithms", fontsize=12)
plt.title("Comparison of Machine Learning Algorithms", fontsize=14, fontweight="bold")

# Rotate x-axis labels
plt.xticks(rotation=20, fontsize=10)

# Add values on bars
for p, acc in zip(ax.patches, accuracy_scores.values()):
    ax.annotate(f"{acc:.2f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 1), ha="center", fontsize=10, fontweight="bold", color="black")

# Adjust y-axis limit dynamically
plt.ylim(0, max(accuracy_scores.values()) + 5)
plt.show()

