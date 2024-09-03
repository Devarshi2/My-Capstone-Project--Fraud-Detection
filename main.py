import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Fraud_Analysis_Dataset.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Select features and target
X = df.drop(columns=['nameOrig', 'nameDest', 'isFraud'])
y = df['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Initialize and train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

dump(clf, "Decision_Tree_Model.joblib", compress=2)