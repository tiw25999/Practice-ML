import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import xgboost as xgb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom Keras Classifier Wrapper
class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y, **fit_params):
        self.model_ = self.build_fn()
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return (self.model_.predict(X) > 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def build_keras_model():
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# โหลดข้อมูลที่ preprocessing แล้ว
df = pd.read_csv(r'C:\Users\Windows 11\Desktop\ML\MLactivity03-03\diabetes_preprocessed.csv')

# แบ่งข้อมูลเป็น Training และ Test set
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
ann = CustomKerasClassifier(build_fn=build_keras_model, epochs=100, batch_size=10, verbose=0)
ann.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
y_pred_ann = ann.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_pred_ann)
print('ANN Accuracy:', accuracy_ann)

# สรุปผลการประเมินโมเดลเบื้องต้น
print("\nสรุปผลการประเมินโมเดลเบื้องต้น:")
print(f'Logistic Regression Accuracy: {accuracy_log}')
print(f'Naive Bayes Accuracy: {accuracy_nb}')
print(f'Decision Tree Accuracy: {accuracy_tree}')
print(f'SVM Accuracy: {accuracy_svm}')
print(f'KNN Accuracy: {accuracy_knn}')
print(f'ANN Accuracy: {accuracy_ann}')

# 6. Hyperparameter Tuning ด้วย GridSearchCV

# Logistic Regression
param_grid_log = {'C': [0.1, 1, 10, 100], 'solver': ['liblinear']}
grid_log = GridSearchCV(LogisticRegression(), param_grid_log, refit=True, verbose=0)
grid_log.fit(X_train, y_train)
best_log = grid_log.best_estimator_
accuracy_log = accuracy_score(y_test, best_log.predict(X_test))
print('Best Logistic Regression Accuracy:', accuracy_log)

# SVM
param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=0)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
accuracy_svm = accuracy_score(y_test, best_svm.predict(X_test))
print('Best SVM Accuracy:', accuracy_svm)

# KNN
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, refit=True, verbose=0)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
accuracy_knn = accuracy_score(y_test, best_knn.predict(X_test))
print('Best KNN Accuracy:', accuracy_knn)

# 7. Cross Validation

# Logistic Regression with Cross Validation
scores_log = cross_val_score(LogisticRegression(C=1, solver='liblinear'), X_scaled, y, cv=10)
print('Logistic Regression Cross-Validation Accuracy:', scores_log.mean())

# SVM with Cross Validation
scores_svm = cross_val_score(SVC(C=1, gamma=0.1, kernel='rbf'), X_scaled, y, cv=10)
print('SVM Cross-Validation Accuracy:', scores_svm.mean())

# KNN with Cross Validation
scores_knn = cross_val_score(KNeighborsClassifier(n_neighbors=3, weights='uniform'), X_scaled, y, cv=10)
print('KNN Cross-Validation Accuracy:', scores_knn.mean())

# 8. การใช้ Ensemble Methods

# Ensemble with Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('lr', LogisticRegression(C=1, solver='liblinear')),
    ('svm', SVC(C=1, gamma=0.1, kernel='rbf', probability=True)),
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='uniform'))
], voting='soft')

ensemble_model.fit(X_train, y_train)
accuracy_ensemble = accuracy_score(y_test, ensemble_model.predict(X_test))
print('Ensemble Model Accuracy:', accuracy_ensemble)

# 9. การใช้ Advanced Models

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Random Forest Accuracy:', accuracy_rf)

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print('Gradient Boosting Accuracy:', accuracy_gb)

# XGBoost
param_distributions_xgb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier()
random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions_xgb, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search_xgb.fit(X_train, y_train)
best_xgb = random_search_xgb.best_estimator_
accuracy_best_xgb = accuracy_score(y_test, best_xgb.predict(X_test))
print('Best XGBoost Accuracy:', accuracy_best_xgb)

# 10. Feature Selection with RFE

# ใช้ Logistic Regression เป็นตัวอย่าง
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X_train, y_train)

print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

# เลือกเฉพาะคุณลักษณะที่ถูกเลือก
X_train_selected = fit.transform(X_train)
X_test_selected = fit.transform(X_test)

# ฝึก Logistic Regression ด้วยคุณลักษณะที่ถูกเลือก
model.fit(X_train_selected, y_train)
y_pred_rfe = model.predict(X_test_selected)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print('Logistic Regression with RFE Accuracy:', accuracy_rfe)

# 11. Stacking

# Stacking Classifier
estimators = [
    ('lr', LogisticRegression(C=1, solver='liblinear')),
    ('svm', SVC(C=1, gamma=0.1, kernel='rbf', probability=True)),
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='uniform'))
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
stacking_model.fit(X_train, y_train)
accuracy_stacking = accuracy_score(y_test, stacking_model.predict(X_test))
print('Stacking Model Accuracy:', accuracy_stacking)

# 12. Data Augmentation

# การเพิ่ม noise ในข้อมูลฝึกอบรม
def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# ใช้ฟังก์ชัน add_noise กับข้อมูลฝึกอบรม
X_train_noisy = add_noise(X_train, noise_level=0.1)

# ฝึกโมเดลด้วยข้อมูลที่เพิ่ม noise
log_reg.fit(X_train_noisy, y_train)
y_pred_log_noisy = log_reg.predict(X_test)
accuracy_log_noisy = accuracy_score(y_test, y_pred_log_noisy)
print('Logistic Regression with Noisy Data Accuracy:', accuracy_log_noisy)

# 13. Feature Engineering with PCA

# ใช้ PCA ลดขนาดคุณลักษณะ
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# ฝึก Logistic Regression ด้วยข้อมูลที่ผ่าน PCA
log_reg.fit(X_train_pca, y_train_pca)
y_pred_pca = log_reg.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print('Logistic Regression with PCA Accuracy:', accuracy_pca)

# สรุปผลการประเมินโมเดลทั้งหมด
print("\nสรุปผลการประเมินโมเดลทั้งหมด:")
print(f'Logistic Regression Accuracy: {accuracy_log}')
print(f'Naive Bayes Accuracy: {accuracy_nb}')
print(f'Decision Tree Accuracy: {accuracy_tree}')
print(f'SVM Accuracy: {accuracy_svm}')
print(f'KNN Accuracy: {accuracy_knn}')
print(f'ANN Accuracy: {accuracy_ann}')
print(f'Best Logistic Regression Accuracy: {accuracy_log}')
print(f'Best SVM Accuracy: {accuracy_svm}')
print(f'Best KNN Accuracy: {accuracy_knn}')
print(f'Logistic Regression Cross-Validation Accuracy: {scores_log.mean()}')
print(f'SVM Cross-Validation Accuracy: {scores_svm.mean()}')
print(f'KNN Cross-Validation Accuracy: {scores_knn.mean()}')
print(f'Ensemble Model Accuracy: {accuracy_ensemble}')
print(f'Random Forest Accuracy: {accuracy_rf}')
print(f'Gradient Boosting Accuracy: {accuracy_gb}')
print(f'Best XGBoost Accuracy: {accuracy_best_xgb}')
print(f'Logistic Regression with RFE Accuracy: {accuracy_rfe}')
print(f'Stacking Model Accuracy: {accuracy_stacking}')
print(f'Logistic Regression with Noisy Data Accuracy: {accuracy_log_noisy}')
print(f'Logistic Regression with PCA Accuracy: {accuracy_pca}')

