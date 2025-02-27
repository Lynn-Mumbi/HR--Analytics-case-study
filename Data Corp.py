import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


aug_train=pd.read_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject\\aug_train.csv\\aug_train.csv")
print(aug_train.head())
aug_test=pd.read_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject\\aug_train.csv\\aug_test.csv")
print(aug_test.shape)

# data preprocessing
print(aug_train.describe())
print(aug_train.shape)
print(aug_train.isna().sum())
print("...........................................................")

# Handling missing values on train set
#Gender
'''On the gender column we had 4508 missing values. Since the data is anonymous I decided to fill the missing values with "other'''
aug_train['gender'].fillna('Other', inplace=True)
# enrolled university
'''This column had 386 missing values. These were replaced by "unknown"'''
aug_train['enrolled_university'].fillna('Unknown', inplace=True)
#education_level
'''This column had 460 missing values replaced by "unknown"'''
aug_train['education_level'].fillna('Unknown', inplace=True)
#major_discipline
"This had 2813 missing values which I filled with the mode "
mode_major_discipline = aug_train['major_discipline'].mode()[0]
aug_train['major_discipline'].fillna(mode_major_discipline, inplace=True)
#experience
'''Experience column had 65 missing values. Since the data is categorical I opted to add another category "unknown".'''
#print(aug_train.info())
print(aug_train["experience"].value_counts())
aug_train['experience'].fillna('unknown', inplace=True)
#company size
'''This column has 5938 missing values that were filled with " unknown" being categorical'''
aug_train['company_size'].fillna('Unknown', inplace=True)
#company type
'''This column had 6140 missing values filled with "unknown"'''
aug_train['company_type'].fillna('Unknown', inplace=True)
#last_new_job
'''This column has 423 missing values filled with " unknonwn'''
aug_train['last_new_job'].fillna('Unknown', inplace=True)

# Validating value counts for each column
for column in aug_train.columns:
    print(f"Value counts for {column}:")
    print(aug_train[column].value_counts())
    print()

#Company size
''' the company size has categories outside scope a 10/49 changed to 10-49 and <10 to 0-9. I decided to  group similar categories together to reduce the number of unique values'''
# Group similar categories
aug_train['company_size'].replace({'10/49': '10-49', '1000-4999': '1000-4999','<10':'0-9'}, inplace=True)

# Validate the updated value counts
print("Updated value counts for company_size:")
print(aug_train['company_size'].value_counts())

print("......................................................................")
# Checking data types
print(aug_train.info())


#handling missing data for test set
aug_test['gender'].fillna('Other', inplace=True)
aug_test['enrolled_university'].fillna('Unknown', inplace=True)
aug_test['education_level'].fillna('Unknown', inplace=True)
aug_test['major_discipline'].fillna(mode_major_discipline, inplace=True)
aug_test['experience'].fillna('unknown', inplace=True)
aug_test['company_size'].fillna('Unknown', inplace=True)
aug_test['company_type'].fillna('Unknown', inplace=True)
aug_test['last_new_job'].fillna('Unknown', inplace=True)
#grouoing similar categories
aug_test['company_size'].replace({'10/49': '10-49', '1000-4999': '1000-4999', '<10': '0-9'}, inplace=True)


#splitting the data into features and target ariables
X_train = aug_train.drop('target', axis=1)
y_train = aug_train['target']
'''When splitting data Features (X_train) and target (y_train) are separated from the training set.'''


# define Xtest
''' We also prepare test set features.'''
X_test = aug_test

#converting the categorical variables to numeric
from sklearn.preprocessing import OneHotEncoder

# Listing categorical features
'''We specify the list of categorical features.'''
categorical_features = ['city', 'gender', 'relevent_experience', 'enrolled_university',
                        'education_level', 'major_discipline','experience', 'company_size', 'company_type', 'last_new_job']
# Listing numerical features
numerical_features = ['city_development_index', 'training_hours']

# Initializing OneHotEncoder
'''We initialize the OneHotEncoder with handle_unknown='ignore' to handle unseen categories during transformation.'''
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


#For Train dataset
'''
We use the OneHotEncoder to encode the categorical features separately for both the training and testing sets.'''
'''''''#encoding  categorical features for train data'''
X_train_encoded = pd.DataFrame(onehot_encoder.fit_transform(X_train[categorical_features]))
'''Update the column names with feature names'''
X_train_encoded.columns = onehot_encoder.get_feature_names_out(categorical_features)

#For test dataset
''' encoding categorical features for test set'''
X_test_encoded = pd.DataFrame(onehot_encoder.transform(X_test[categorical_features]))
''' updating  column names with feature names'''
X_test_encoded.columns = onehot_encoder.get_feature_names_out(categorical_features)

''' concatenate encoded features with numerical features for training and test.
    we concatinate the encoded categorical features with the numerical features.'''
X_train_processed = pd.concat([X_train_encoded, X_train[numerical_features].reset_index(drop=True)], axis=1)
X_test_processed = pd.concat([X_test_encoded, X_test[numerical_features].reset_index(drop=True)], axis=1)

''' we then  ensure that the columns match after encoding'''
X_train_processed = X_train_processed.loc[:, ~X_train_processed.columns.duplicated()]
X_test_processed = X_test_processed.loc[:, ~X_test_processed.columns.duplicated()]

#print(X_train_processed.head())

#TRAINING
'''Training the Model: A RandomForestClassifier is initialized and trained on the processed training data.
I chose this because it is a robust, versatile algorithm that typically performs well on a variety of datasets.
'''
# Initialize and train using the  RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_processed, y_train)

# Since we do not have the target variable for the test set, we cannot evaluate on the test set directly.
# However, we can make predictions and save them for submission or further analysis.

# Predict on the test set
y_test_pred = rf_classifier.predict(X_test_processed)

# Output the predictions
test_predictions = pd.DataFrame({'enrollee_id': aug_test['enrollee_id'], 'target': y_test_pred})
print(test_predictions[test_predictions['target']==1.0])



#visualizations on the catego
# Bar plots for categorical variables
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
sns.countplot(x='gender', data=aug_train)
plt.title('Gender Distribution')

plt.subplot(2, 3, 2)
sns.countplot(x='relevent_experience', data=aug_train)
plt.title('Relevant Experience Distribution')

plt.subplot(2, 3, 3)
sns.countplot(x='enrolled_university', data=aug_train)
plt.title('Enrolled University Distribution')

plt.subplot(2, 3, 4)
sns.countplot(x='education_level', data=aug_train)
plt.title('Education Level Distribution')

plt.subplot(2, 3, 5)
sns.countplot(x='major_discipline', data=aug_train)
plt.title('Major Discipline Distribution')

plt.subplot(2, 3, 6)
sns.countplot(x='company_size', data=aug_train)
plt.title('Company Size Distribution')

plt.tight_layout()
plt.show()


# Box plots for numerical variables across different categories
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='education_level', y='training_hours', data=aug_train)
plt.title('Training Hours by Education Level')

plt.subplot(1, 2, 2)
sns.boxplot(x='company_size', y='city_development_index', data=aug_train)
plt.title('City Development Index by Company Size')

plt.tight_layout()
plt.show()

# Count plot for target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=test_predictions)
plt.title('Target Class Distribution')
plt.show()


#################################################################################
#choosing the best model
'''ensure necessary imports'''
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize the models
'''to choose the best model we put the models we would like to test in a dictionary and loop through each'''
'''models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Dictionary to store the results
results = {}

#Training and evaluate each model
for model_name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    y_pred_prob = model.predict_proba(X_test_processed)[:, 1]

    # We don't have y_test, so let's assume the same processing for validation on train set split
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)
    model.fit(X_train_part, y_train_part)
    y_val_pred = model.predict(X_val)
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred_prob)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)'''

# For actual prediction on the test set, use the best model identified from validation
'''since processing took time here I opted for random forest classifier'''
'''best_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Replace with the best model found
best_model.fit(X_train_processed, y_train)
y_test_pred = best_model.predict(X_test_processed)

# Output the predictions
test_predictions = pd.DataFrame({'enrollee_id': aug_test['enrollee_id'], 'target': y_test_pred})
print(test_predictions.head())

'''

print(aug_test.shape)






