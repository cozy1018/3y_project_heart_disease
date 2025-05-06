# With smote (LR)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

a = 59
# Load and preprocess the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df['num'] = df['num'].astype('str')

df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# Convert boolean columns to integers
for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype(int)

# Define features (X) and the target variable (y)
X = df.drop('num_1', axis=1)
X = X.drop('id', axis=1)
y = df['num_1']

# Function to identify the least important feature
def get_least_important_feature(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=a)
    feature_importances = result.importances_mean
    least_important_index = np.argmin(feature_importances)
    return least_important_index

# Function to train and evaluate Logistic Regression with grid search, 5-fold CV, and permutation importance
def train_and_evaluate_logistic_with_gridsearch(X, y, cv=5):
    num_features = []
    accuracies = []
    features_used = []
    test_accuracies = []
    remaining_features = list(X.columns)

    # Define parameter grid for GridSearchCV
    param_grid = [
        {'penalty': ['l1','l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,
                                       13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['liblinear']},  
        {'penalty': ['l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,
                                  13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['lbfgs', 'sag']} 
    ]

    while len(remaining_features) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X[remaining_features], y, test_size=0.2, random_state=a)

        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=a)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        scaler_smote = StandardScaler()
        X_train_resampled_scaled = scaler_smote.fit_transform(X_train_resampled)
        X_test_scaled = scaler_smote.transform(X_test)

        # Using Logistic Regression with GridSearchCV for hyperparameter tuning
        model = LogisticRegression(random_state=a, max_iter=1000)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_search.fit(X_train_resampled_scaled, y_train_resampled)

        # Get best parameters from grid search
        best_params = grid_search.best_params_
        print(f"Best parameters for this iteration: {best_params}")

        # Predict and evaluate accuracy using cross-validation
        best_model = grid_search.best_estimator_
        cv_accuracy = grid_search.best_score_
        y_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Store results
        num_features.append(len(remaining_features))
        accuracies.append(cv_accuracy)
        test_accuracies.append(test_accuracy)
        features_used.append(list(remaining_features))  

        # Identify and remove the least important feature
        least_important_index = get_least_important_feature(best_model, X_test_scaled, y_test)
        remaining_features.pop(least_important_index)

    return num_features, accuracies, features_used, test_accuracies

# With SMOTE
num_features_smote, accuracies_smote, features_used_smote, test_accuracies_smote = train_and_evaluate_logistic_with_gridsearch(
    X, y
)

# Plot accuracy vs. number of features for the SMOTE scena': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}rio
plt.figure(figsize=(8, 6))
plt.plot(num_features_smote, test_accuracies_smote, marker='o', linestyle='-', color='g')
plt.xlabel('Number of Features')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Number of Features (Logistic Regression with GridSearchCV, with SMOTE)')
plt.grid()
plt.show()

# Print results
print("\nFinal Results with SMOTE:")
for n, acc, features, test_acc in zip(num_features_smote, accuracies_smote, features_used_smote, test_accuracies_smote):
    print(f"Number of features: {n}, Cross-Validation Accuracy: {acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Features used: {features}")
