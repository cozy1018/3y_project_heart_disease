#################
#################
# Without smote

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

def generateRank1(score, class_no):
    rank = np.zeros([class_no, 1])
    scores = score
    for i in range(class_no):
        rank[i] = 1 - np.exp(-((scores[i] - 1) ** 2) / 2.0)
    return rank

def generateRank2(score, class_no):
    rank = np.zeros([class_no, 1])
    scores = score
    for i in range(class_no):
        rank[i] = 1 - np.tanh(((scores[i] - 1) ** 2) / 2)
    return rank

def doFusion(res1, res2, res3, label, class_no):
    cnt = 0
    id = []
    for i in range(len(res1)):
        rank1 = generateRank1(res1[i], class_no) * generateRank2(res1[i], class_no)
        rank2 = generateRank1(res2[i], class_no) * generateRank2(res2[i], class_no)
        rank3 = generateRank1(res3[i], class_no) * generateRank2(res3[i], class_no)
        rankSum = rank1 + rank2 + rank3
        scoreSum = 1 - (res1[i] + res2[i] + res3[i]) / 3
        fusedScore = (rankSum.T) * scoreSum
        cls = np.argmin(rankSum)
        
        if ((cls < class_no) and (cls==label[i])):
            cnt += 1
        id.append(cls)
    #print(cnt / len(res1))
    return id

# Load health dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]  # Filter for Cleveland dataset
df = df[df['age'] != 28]  # Remove specific age for this dataset

# Convert target values: keep 0 as 0 and group 1-4 into class 1 (heart disease)
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df['num'] = df['num'].astype('str')

# Drop rows with missing values
df = df.dropna()

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop('num_1', axis=1)  
X = X.drop('id', axis=1)  
y = df['num_1']  

num_of_classes = 2 
rs = 42

# Define models for the ensemble
svm_model = SVC(probability=True, random_state=rs)
lr_model = LogisticRegression(max_iter=1000)
nb_model = GaussianNB()

# Store predictions for fusion
pred_svm = []
pred_lr = []
pred_nb = []
actual = []

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
svm_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)
nb_model.fit(X_train_scaled, y_train)

# Make predictions
res_svm = svm_model.predict_proba(X_test_scaled)
res_lr = lr_model.predict_proba(X_test_scaled)
res_nb = nb_model.predict_proba(X_test_scaled)

# Collect predictions
pred_svm.append(res_svm)
pred_lr.append(res_lr)
pred_nb.append(res_nb)
actual.append(y_test)
    
# Flatten predictions for fusion
pred_svm = np.concatenate(pred_svm, axis=0)
pred_lr = np.concatenate(pred_lr, axis=0)
pred_nb = np.concatenate(pred_nb, axis=0)
actual = np.concatenate(actual, axis=0)

# Apply fuzzy rank-based ensemble learning
ensemble_preds = doFusion(pred_svm, pred_lr, pred_nb, actual, num_of_classes)

# Calculate accuracy for the ensemble model
accuracy = accuracy_score(actual, ensemble_preds)
print(f"Accuracy of Ensemble Model (Fuzzy Rank-Based): {accuracy:.4f}")