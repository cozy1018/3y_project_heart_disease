# MOE [without smote]

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load and process the data
df = pd.read_csv("datafile/heart_disease_uci.csv")  
df = df[df['dataset'] == "Cleveland"]  
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df['num'] = df['num'].astype('str')

# Drop rows with missing values
df = df.dropna()  

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop(['num_1', 'id'], axis=1)
y = df['num_1']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train expert models (SVM, Logistic Regression, Naive Bayes)
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Generate probabilities from the expert models
svm_pred_train = svm.predict_proba(X_train_scaled)[:, 1]  # Probability of class 1 
lr_pred_train = lr.predict_proba(X_train_scaled)[:, 1]    
nb_pred_train = nb.predict_proba(X_train_scaled)[:, 1]    

svm_pred_test = svm.predict_proba(X_test_scaled)[:, 1]  
lr_pred_test = lr.predict_proba(X_test_scaled)[:, 1]    
nb_pred_test = nb.predict_proba(X_test_scaled)[:, 1]    

# Combine the predictions to create new feature set for deep learning model
X_train_experts = np.vstack([svm_pred_train, lr_pred_train, nb_pred_train]).T
X_test_experts = np.vstack([svm_pred_test, lr_pred_test, nb_pred_test]).T

# Build and train the deep learning model using Keras
model = Sequential()
model.add(Input(shape=(3,))) 
model.add(Dense(1, activation='sigmoid'))  
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_experts, y_train, epochs=400, batch_size=32, verbose=0)

# Make predictions and evaluate the model
predictions = model.predict(X_test_experts)
predictions = (predictions > 0.5).astype(int)  

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy of the ensemble model: {accuracy * 100:.2f}%')

# Calculate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')