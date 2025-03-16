# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:32:26 2025

@author: mmmkh
"""

# Mounting Google Drive in Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Importing Libraries for Data Processing and Classification Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Reading a CSV File from Google Drive and Skipping Bad Lines
df = pd.read_csv('/content/drive/MyDrive/data.csv', engine='python', on_bad_lines='skip')

# Setting Column Names in DataFrame
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_hot_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "normal."
]
df.columns = column_names

# Displaying Data Information
print("Data Info:")
df.info()

# Calculating Missing Values in the Data
missing_data = df.isnull().sum()
print("Missing Values:\n", missing_data)

# Identifying and Removing Duplicate Rows from Data
print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")

# Visualizing Value Distribution for Categorical Columns
categorical_columns = ['protocol_type', 'service', 'flag']
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    df[col].value_counts().plot(kind='bar', color='blue')
    plt.title(f"Distribution of {col}")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

# Encoding categorical variables
label_encoder = LabelEncoder()
df['protocol_type'] = label_encoder.fit_transform(df['protocol_type'])
df['flag'] = label_encoder.fit_transform(df['flag'])
df['normal.'] = label_encoder.fit_transform(df['normal.'])

# Replacing Values in 'service' Column with Frequency of Each Category
service_frequency = df['service'].value_counts()
df['service'] = df['service'].map(service_frequency)

# Defining Features (X) and Target (y) from Data
X = df.iloc[:, :-1]
y = df['normal.']

# Balancing Data Using SMOTE for Imbalanced Data
smote = SMOTE(random_state=42, k_neighbors=1)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Splitting Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} Confusion Matrix:")
    print(conf_matrix)
    print(f"{model_name} Accuracy: {accuracy:.4f}\n")
    
    return accuracy

# Building, Training, and Evaluating Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='linear')
}

for model_name, model in models.items():
    evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

# Visualizing Correlation Heatmap of Variables
plt.figure(figsize=(22, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Displaying Distribution of All Columns Using Histograms
plt.figure(figsize=(30, 20))
df.hist(figsize=(30, 20), bins=20)
plt.show()