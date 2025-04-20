#!/usr/bin/env python
# coding: utf-8

# # A TikTok video classification (ref. Google-Coursera TikTok project)
# # Hyperparameter Tuning using GridSearchCV

# In[1]:


# Import the necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('c:/Users/Windows 11/Desktop/ML/Ac06/tiktok_dataset.csv')

# Open a file to write the output
with open('output.txt', 'w') as f:
    # Display the first five rows of the dataset
    f.write("First five rows of the dataset:\n")
    f.write(df.head(5).to_string() + '\n\n')
    
    # Display descriptive statistics
    f.write("Descriptive statistics before preprocessing:\n")
    f.write(df.describe().to_string() + '\n\n')

    # # Preprocess the dataset
    # Drop the '#' and 'video_id' columns
    df = df.drop(['#', 'video_id'], axis=1)
    # Handling missing values
    df = df.dropna()
    # Handling duplicate values
    df = df.drop_duplicates()
    # Handling outliers
    df = df[(df['video_download_count'] > 0) & (df['video_like_count'] > 0) & (df['video_comment_count'] > 0) & (df['video_share_count'] > 0)]

    # Display descriptive statistics of the dataset after preprocessing
    f.write("Descriptive statistics after preprocessing:\n")
    f.write(df.describe().to_string() + '\n\n')
    
    # Check class balance
    f.write("Class balance:\n")
    f.write(df['claim_status'].value_counts().to_string() + '\n\n')

    # # Feature Engineering
    # Extract the length of each video_transcription_text and add this as a column to the dataframe,
    # so that it can be used as a potential feature in the model.
    df['video_transcription_text_length'] = df['video_transcription_text'].apply(lambda x: len(x))
    df.drop('video_transcription_text', axis=1, inplace=True)

    # Calculate the average text_length for claims and opinions.
    average_text_length_claim = df[df['claim_status'] == 'claim']['video_transcription_text_length'].mean()
    average_text_length_opinion = df[df['claim_status'] == 'opinion']['video_transcription_text_length'].mean()

    # Encode target and categorical variables.
    from sklearn.preprocessing import LabelEncoder

    mapping = {}
    # Encode the categorical variables
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Display the mapping of the categorical variables
    f.write("Mapping of categorical variables:\n")
    f.write(str(mapping) + '\n\n')

    # Display descriptive statistics of the dataset after feature engineering
    f.write("Descriptive statistics after feature engineering:\n")
    f.write(df.describe(include='all').to_string() + '\n\n')


# In[2]:


# # Prepare the data for modeling
# Split the dataset into features and target variable
X = df.drop('claim_status', axis=1)
y = df['claim_status']

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Open the file again in append mode
with open('output.txt', 'a') as f:
    # Display the shape of the training and testing sets
    f.write('Training set shape: {}\n'.format(X_train.shape))
    f.write('Testing set shape: {}\n\n'.format(X_test.shape))


# In[3]:


# # Train the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# # GridSearchCV
from sklearn.model_selection import GridSearchCV

# scoring metrics
scoring = {'accuracy': 'accuracy',
           'precision': 'precision_weighted',
           'recall': 'recall_weighted',
           'f1': 'f1_weighted'}

# Create the hyperparameter grid - search space
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'max_samples': [0.5, 0.7, 0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, 
                           verbose=2, scoring=scoring, refit='accuracy')

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Open the file again in append mode
with open('output.txt', 'a') as f:
    # Display the best parameters
    f.write('Best Parameters: {}\n\n'.format(best_params))
    
    # Display the best score
    f.write('Best Score: {}\n\n'.format(grid_search.best_score_))


# In[4]:


# # Evaluate the model

# Get the best model
best_model = grid_search.best_estimator_

# Predict the test set
y_pred = best_model.predict(X_test)

# Calculate the classification report
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Open the file again in append mode
with open('output.txt', 'a') as f:
    f.write('Classification Report:\n')
    f.write(classification_report(y_test, y_pred) + '\n')

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    f.write('Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}\n\n'.format(accuracy, precision, recall, f1))


# In[5]:


# Visualize the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 5))
sns.set(font_scale=1.5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
