import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    # Load the dataset
    df = pd.read_csv(r'C:\Users\Windows 11\Desktop\ML\Mo7\patient_diagnosis.csv')

    # Display the first five rows of the dataset
    print(df.head(5).to_string(), '\n')
    # Display descriptive statistics
    print(df.describe(include='all').to_string(), '\n')

    # Preprocess the dataset
    df = df.dropna()  # Handling missing values
    df = df.drop_duplicates()  # Handling duplicate values

    # Check class balance
    print(df['Outcome Variable'].value_counts().to_string(), '\n')

    n = 1
    plt.figure(figsize=(20, 10))
    sns.set(font_scale=1.5)
    for i in df.drop('Disease', axis=1).columns:
        plt.subplot(3, 5, n)
        if df[i].dtype == 'object':
            sns.countplot(y=df[i])
        else:
            sns.kdeplot(df[i])
            plt.grid()
        n += 1
    plt.tight_layout()
    plt.title('Feature Distribution')
    plt.show()

    # Encode target and categorical variables
    mapping = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Display the mapping of the categorical variables
    print(mapping, '\n')

    # Display descriptive statistics of the dataset after preprocessing
    print(df.describe(include='all').to_string(), '\n')

    # Prepare the data for modeling
    X = df.drop('Outcome Variable', axis=1)
    y = df['Outcome Variable']

    # K-fold cross-validation with stratification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Display the shape of the training and testing sets
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape, '\n')

    # Train Model using RandomForestClassifier
    rf = RandomForestClassifier()

    # GridSearchCV
    param_grid_rf = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', refit=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate Model
    y_pred = best_model.predict(X_test)

    print('Random Forest Classification Report:')
    print(classification_report(y_test, y_pred), '\n')

    accuracy = accuracy_score(y_test, y_pred)
    presicion = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('Random Forest Accuracy: {0}, Precision: {1}, Recall: {2}, F1 Score: {3}'.format(accuracy, presicion, recall, f1), '\n')

    # Save the model
    model_dir = r'C:\Users\Windows 11\Desktop\ML\Mo7\models'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(mapping, os.path.join(model_dir, 'mapping.pkl'))
    joblib.dump(X.columns, os.path.join(model_dir, 'columns.pkl'))

if __name__ == '__main__':
    main()
