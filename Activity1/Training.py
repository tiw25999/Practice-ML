import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# โหลดข้อมูลที่ preprocessing แล้ว
df = pd.read_csv(r'C:\Users\Windows 11\Desktop\ML\MLactivity03-03\diabetes_preprocessed.csv')

# แบ่งข้อมูลเป็น Training และ Test set
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. การสร้างและฝึกโมเดลทั้ง 6 แบบ

# 5.1 Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print('Logistic Regression Accuracy:', accuracy_log)

# 5.2 Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print('Naive Bayes Accuracy:', accuracy_nb)

# 5.3 Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print('Decision Tree Accuracy:', accuracy_tree)

# 5.4 Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print('SVM Accuracy:', accuracy_svm)

# 5.5 K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print('KNN Accuracy:', accuracy_knn)

# 5.6 Artificial Neural Network (ANN)
ann = Sequential()
ann.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
ann.add(Dense(8, activation='relu'))
ann.add(Dense(16, activation='relu'))
ann.add(Dense(32, activation='relu'))
ann.add(Dense(64, activation='relu'))
ann.add(Dense(128, activation='relu'))
ann.add(Dense(256, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))

ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=100, batch_size=20, verbose=0)
y_pred_ann = (ann.predict(X_test) > 0.5).astype(int)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
print('ANN Accuracy:', accuracy_ann)

# สรุปผลการประเมินโมเดล
print("\nสรุปผลการประเมินโมเดล:")
print(f'Logistic Regression Accuracy: {accuracy_log}')
print(f'Naive Bayes Accuracy: {accuracy_nb}')
print(f'Decision Tree Accuracy: {accuracy_tree}')
print(f'SVM Accuracy: {accuracy_svm}')
print(f'KNN Accuracy: {accuracy_knn}')
print(f'ANN Accuracy: {accuracy_ann}')
