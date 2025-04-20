import pandas as pd

# 1. โหลดและตรวจสอบข้อมูล
file_path = r'C:\Users\Windows 11\Desktop\ML\MLactivity03-03\diabetes.csv'
df = pd.read_csv(file_path)
print("ข้อมูลเบื้องต้น:")
print(df.head())

# 2. ตรวจสอบข้อมูลที่ขาดหาย
print("\nการตรวจสอบค่าว่าง:")
print(df.isnull().sum())

# 3. การจัดการกับค่าว่าง (ถ้ามี)
# แทนที่ค่า 0 ในคอลัมน์ที่ควรมีค่าที่ไม่เป็น 0
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, pd.NA)

# แทนที่ค่า NA ด้วยค่ามัธยฐานของแต่ละคอลัมน์
df.fillna(df.median(), inplace=True)

print("\nข้อมูลหลังจากการจัดการค่าว่าง:")
print(df.head())

# 4. การบันทึกข้อมูลที่ preprocessing เสร็จแล้วเป็นไฟล์ใหม่
output_file_path = r'C:\Users\Windows 11\Desktop\ML\MLactivity03-03\diabetes_preprocessed.csv'
df.to_csv(output_file_path, index=False)
print(f"\nบันทึกไฟล์ที่ preprocessing เสร็จแล้วเป็นไฟล์ใหม่ที่ {output_file_path}")
