# Bagging - LR, SVM, Naïve Bayes

# Import necessary libraries 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier

print("Processing data with two classes to build models, applying feature scaling, one-hot encoding, and bagging ensemble.")

# Load and filter the dataset
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

# Define features (X) and the target variable (y)
X = df.drop(['num_1', 'id'], axis=1, errors='ignore')
y = df['num_1']
rs = 42

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rs
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base estimators: SVM, Logistic Regression, and Naïve Bayes
svm = SVC(random_state=rs, probability=True)  # Ensure probability=True for compatibility
lr = LogisticRegression(random_state=rs, max_iter=1000)
nb = GaussianNB()

# Combine the classifiers into a VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('nb', nb)], 
    voting='hard'
)

# Use BaggingClassifier with the VotingClassifier as the base estimator
bagging_voting = BaggingClassifier(
    estimator=voting_clf,
    random_state=rs
)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],  # Number of models to train
    'max_samples': [0.8, 0.9, 1.0],  # Fraction of training samples
}

# Perform grid search
grid_search = GridSearchCV(bagging_voting, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
