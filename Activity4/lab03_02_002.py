import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# ฟังก์ชันทดสอบโมเดล
def test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)  # ฝึกโมเดลด้วยข้อมูล
    y_pred = model.predict(X_test)  # ทำนายผลด้วยข้อมูลทดสอบ
    print(classification_report(y_test, y_pred))  # แสดงรายงานการทำนาย
    print("ความแม่นยำ:", accuracy_score(y_test, y_pred))  # แสดงค่าความแม่นยำ

# โหลดข้อมูลฝึกและทดสอบที่ผ่านการ preprocessing แล้ว
train_file_path = 'C:\\Users\\Windows 11\\Desktop\\CoronaML\\preprocessed_Corona_NLP_train.csv'
test_file_path = 'C:\\Users\\Windows 11\\Desktop\\CoronaML\\preprocessed_Corona_NLP_test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# แยกฟีเจอร์และค่าเป้าหมาย
X_train = train_data.drop('Sentiment', axis=1)
y_train = train_data['Sentiment']
X_test = test_data.drop('Sentiment', axis=1)
y_test = test_data['Sentiment']

# สร้างโมเดล Logistic Regression พร้อมกับ Grid Search
param_grid = {'C': [1, 10, 100, 1000, 3792]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# ทดสอบโมเดลที่ดีที่สุด
best_model = grid_search.best_estimator_
test_model(best_model, X_train, y_train, X_test, y_test)