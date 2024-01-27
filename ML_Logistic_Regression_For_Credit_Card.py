import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load features (X) from the first CSV file
# Replace 'FeaturesDataset.csv' with the actual filename or URL of your features dataset
features_df = pd.read_csv("https://raw.githubusercontent.com/pujan08/ML_Logistic_Regression_For_Credit_Card/main/Credit_card.csv")
string_imputer = SimpleImputer(strategy='constant', fill_value='missing')
features_df_imputed = pd.DataFrame(string_imputer.fit_transform(features_df), columns=features_df.columns)
string_imputer.fit(features_df)
missing_values_after_imputation = features_df_imputed.isna().sum().sum()
print(features_df_imputed)

if missing_values_after_imputation == 0:
    print("Sanity Check Passed: No missing values after imputation in features_df.")
else:
    print(f"Sanity Check Failed: There are still {missing_values_after_imputation} missing values in features_df after imputation.")


# Load target variable (y) from the second CSV file
# Replace 'TargetDataset.csv' with the actual filename or URL of your target variable dataset

target_df = pd.read_csv("https://raw.githubusercontent.com/pujan08/ML_Logistic_Regression_For_Credit_Card/main/Credit_card_label.csv")

numeric_imputer = SimpleImputer(strategy='mean')
target_df_imputed = pd.DataFrame(numeric_imputer.fit_transform(target_df), columns=target_df.columns)
numeric_imputer.fit(target_df)
missing_values_after_imputation = target_df_imputed.isna().sum().sum()
print(target_df_imputed)

if missing_values_after_imputation == 0:
    print("Sanity Check Passed: No missing values after imputation in target_df.")
else:
    print(f"Sanity Check Failed: There are still {missing_values_after_imputation} missing values in target_df after imputation.")

# Assuming both datasets have a common column (e.g., 'ID') to merge on
merged_df = pd.merge(features_df_imputed, target_df_imputed, on='Ind_ID', how='inner')


# Define Features (X) and Target Variable (y)
X = merged_df[['Car_Owner','Propert_Owner','Annual_income','Employed_days','Family_Members']]
X = pd.get_dummies(X, columns=['Car_Owner','Propert_Owner','Annual_income','Employed_days','Family_Members'])
y = merged_df['label']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess numerical features (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Logistic Regression model
model = LogisticRegression()
# Define the parameter grid for Logistic Regression
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear']
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and corresponding mean cross-validated score
print("Best Parameters:", grid_search.best_params_)
print("Best Mean Accuracy:", grid_search.best_score_)

# Make predictions on the test data using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
