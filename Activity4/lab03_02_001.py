import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ฟังก์ชันทำความสะอาดข้อความ
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # ลบลิงก์
    text = re.sub(r'@\w+', '', text)  # ลบการกล่าวถึง (@mention)
    text = re.sub(r'#\w+', '', text)  # ลบแฮชแท็ก (#hashtag)
    text = re.sub(r'\d+', '', text)  # ลบตัวเลข
    text = re.sub(r'[^\w\s]', '', text)  # ลบอักขระพิเศษ
    text = re.sub(r'\s+', ' ', text)  # ลบช่องว่างที่ไม่จำเป็น
    return text.strip()

# ฟังก์ชันเพิ่มฟีเจอร์
def add_empirical_features(data):
    data['tweet_length'] = data['OriginalTweet'].apply(len)
    data['word_count'] = data['OriginalTweet'].apply(lambda x: len(x.split()))
    data['sentence_count'] = data['OriginalTweet'].apply(lambda x: len(re.split(r'[.!?]+', x)) - 1)
    return data[['tweet_length', 'word_count', 'sentence_count']]

# ฟังก์ชันโหลดและเตรียมข้อมูล
def load_and_prepare_data(file_path, is_train=True):
    data = pd.read_csv(file_path, encoding='latin1')  # โหลดข้อมูลจากไฟล์ CSV ด้วย encoding 'latin1'
    data.dropna(subset=['OriginalTweet'], inplace=True)  # ลบแถวที่มีค่าหายไปในคอลัมน์ 'OriginalTweet'
    if is_train:
        data.dropna(subset=['Sentiment'], inplace=True)  # ลบแถวที่มีค่าหายไปในคอลัมน์ 'Sentiment'
    sentiment_mapping = {'Extremely Positive': 2, 'Positive': 1, 'Neutral': 0, 'Negative': -1, 'Extremely Negative': -2}  # mapping ค่าของ Sentiment เป็นตัวเลข
    data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)  # แปลงค่า Sentiment เป็นตัวเลข
    data['OriginalTweet'] = data['OriginalTweet'].apply(clean_text).str.lower()  # ทำความสะอาดข้อความและแปลงเป็นตัวพิมพ์เล็กทั้งหมด
    return data

# ฟังก์ชันแปลงข้อความเป็น TF-IDF features
def transform_to_tfidf(train_data, test_data):
    vectorizer = TfidfVectorizer(
        max_features=1000,  # กำหนดจำนวนฟีเจอร์สูงสุดเป็น 1000
        ngram_range=(1, 2),  # ใช้ทั้ง 1-grams และ 2-grams
        norm='l2',  # ใช้การนอร์ม L2
        use_idf=True,  # ใช้ IDF
        smooth_idf=True,
        sublinear_tf=True,
        min_df=2,  # คำต้องปรากฏอย่างน้อยใน 2 เอกสาร
        max_df=0.5  # คำต้องปรากฏไม่เกิน 50% ของเอกสารทั้งหมด
    )
    X_train = vectorizer.fit_transform(train_data['OriginalTweet'])  # แปลงข้อความในข้อมูลฝึกเป็น TF-IDF features
    X_test = vectorizer.transform(test_data['OriginalTweet'])  # แปลงข้อความในข้อมูลทดสอบเป็น TF-IDF features
    return X_train, X_test, vectorizer

# ฟังก์ชันลดมิติของข้อมูลโดยใช้ PCA
def apply_pca(X_train, X_test, n_components=100):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train.toarray())  # ลดมิติของข้อมูลฝึกโดยใช้ PCA
    X_test_pca = pca.transform(X_test.toarray())  # ลดมิติของข้อมูลทดสอบโดยใช้ PCA
    return X_train_pca, X_test_pca

# ฟังก์ชันปรับขนาดข้อมูลให้อยู่ในช่วง 0-1
def scale_data(X_train, X_test):
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ฟังก์ชันบันทึกข้อมูลลงไฟล์ CSV
def save_to_csv(X, y, output_path):
    df = pd.DataFrame(X)  # สร้าง DataFrame จาก TF-IDF features ที่ถูกลดมิติแล้ว
    if y is not None:
        df['Sentiment'] = y.values  # เพิ่มคอลัมน์ 'Sentiment' ลงใน DataFrame
    df.to_csv(output_path, index=False)  # บันทึก DataFrame ลงในไฟล์ CSV โดยไม่ใส่ index
    print(f'TF-IDF features และค่าที่ได้ถูกบันทึกใน {output_path}')  # แสดงข้อความยืนยันการบันทึกไฟล์

# เส้นทางไฟล์ข้อมูลและไฟล์ผลลัพธ์
train_file_path = r'C:\Users\Windows 11\Desktop\CoronaML\Corona_NLP_train.csv'
test_file_path = r'C:\Users\Windows 11\Desktop\CoronaML\Corona_NLP_test.csv'
train_output_path = r'C:\Users\Windows 11\Desktop\CoronaML\preprocessed_Corona_NLP_train.csv'
test_output_path = r'C:\Users\Windows 11\Desktop\CoronaML\preprocessed_Corona_NLP_test.csv'

# โหลดและเตรียมข้อมูล
train_data = load_and_prepare_data(train_file_path, is_train=True)
test_data = load_and_prepare_data(test_file_path, is_train=False)

# เพิ่มฟีเจอร์
train_empirical_features = add_empirical_features(train_data)
test_empirical_features = add_empirical_features(test_data)

# ตรวจสอบการกระจายของคำในข้อความฝึก
all_text = ' '.join(train_data['OriginalTweet'])
word_counts = Counter(all_text.split())
print("คำที่พบมากที่สุด 10 อันดับ:")
print(word_counts.most_common(10))

# ดูตัวอย่างข้อความ
print("ข้อความตัวอย่าง:")
print(train_data['OriginalTweet'].head())

# แปลงข้อความเป็น TF-IDF features
X_train_tfidf, X_test_tfidf, vectorizer = transform_to_tfidf(train_data, test_data)
# ลดมิติของข้อมูลโดยใช้ PCA
X_train_pca, X_test_pca = apply_pca(X_train_tfidf, X_test_tfidf)

# รวมฟีเจอร์กับ TF-IDF features ที่ผ่านการลดมิติแล้ว
X_train_combined = pd.concat([pd.DataFrame(X_train_pca), train_empirical_features.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([pd.DataFrame(X_test_pca), test_empirical_features.reset_index(drop=True)], axis=1)
# ปรับขนาดข้อมูลให้อยู่ในช่วง 0-1
X_train_scaled, X_test_scaled = scale_data(X_train_combined, X_test_combined)

# ตรวจสอบจำนวนฟีเจอร์ที่ถูกสร้างขึ้น
print("จำนวนฟีเจอร์ที่ถูกสร้างขึ้น:", len(vectorizer.get_feature_names_out()))

# แสดงตัวอย่างของ TF-IDF features และฟีเจอร์
print("TF-IDF features ตัวอย่าง (หลังจากทำ PCA และการปรับขนาด) ในข้อมูล:")
print(X_train_scaled[:5])
print("บันทึก Sentiment:")
print(train_data['Sentiment'].head())

# บันทึก TF-IDF features และค่าที่ได้ลงในไฟล์ CSV
save_to_csv(X_train_scaled, train_data['Sentiment'], train_output_path)
save_to_csv(X_test_scaled, test_data['Sentiment'], test_output_path)
