import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าฟอนต์เริ่มต้น
plt.rcParams['font.family'] = 'Tahoma'

# โหลดชุดข้อมูล
dataset = pd.read_csv(r'C:\Users\Windows 11\Desktop\dataAnn\Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# เข้ารหัส 'Gender'
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# เข้ารหัส 'Geography'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# แบ่งชุดข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดฟีเจอร์
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# สร้างโมเดล ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=125, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=75, activation='relu'))
ann.add(tf.keras.layers.Dense(units=125, activation='relu'))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
ann.add(tf.keras.layers.Dense(units=75, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# คอมไพล์โมเดล ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล ANN
ann.fit(X_train, y_train, batch_size=32, epochs=300)

# งานที่ 1: สร้างข้อมูลทดสอบ 5 ชุดที่แตกต่างกันเพื่อทำการทำนายผล
test_samples = [
    [1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000],
    [0, 1, 0, 500, 0, 35, 5, 80000, 1, 1, 0, 75000],
    [0, 0, 1, 700, 1, 50, 10, 120000, 3, 0, 1, 90000],
    [1, 0, 0, 450, 0, 29, 2, 50000, 2, 1, 1, 30000],
    [0, 1, 0, 670, 1, 42, 8, 70000, 1, 0, 0, 65000]
]

# ปรับขนาดข้อมูลทดสอบ
test_samples = sc.transform(test_samples)

# ทำนายผลโดยใช้ข้อมูลทดสอบ
predictions = ann.predict(test_samples)
predictions = predictions > 0.5

# แสดงผลการทำนายข้อมูลทดสอบ
print("ผลการทำนายสำหรับตัวอย่างทดสอบ:", predictions)

# งานที่ 2: ทำนายผลชุดทดสอบ
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
results = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# แสดงผลการทำนายชุดทดสอบ
print("ผลการทำนายสำหรับชุดทดสอบ:\n", pd.DataFrame(results, columns=["ทำนาย", "จริง"]))

# งานที่ 3: สร้าง Confusion Matrix และ Accuracy Score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# แสดง Confusion Matrix และ Accuracy Score
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy)

# การแสดง Confusion Matrix ด้วย Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ทำนายว่าไม่', 'ทำนายว่าใช่'], yticklabels=['จริงว่าไม่', 'จริงว่าใช่'])
plt.xlabel('ทำนาย')
plt.ylabel('จริง')
plt.title('Confusion Matrix')
plt.show()
