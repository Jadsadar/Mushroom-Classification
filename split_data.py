import os
import shutil
from sklearn.model_selection import train_test_split

# ตั้งค่าโฟลเดอร์
input_folder = "resource\mush_data"  # โฟลเดอร์หลักที่เก็บโฟลเดอร์ย่อยของเห็ดพิษและเห็ดกินได้
output_folder = "split_data"  

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ตรวจสอบว่าค่า ratio รวมกันได้ 1.0
assert train_ratio + val_ratio + test_ratio == 1.0, "ค่า ratio ต้องรวมกันได้ 1.0"

# สร้างโฟลเดอร์ Train, Val, Test
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

# วนลูปตามแต่ละหมวดหมู่ (เห็ดกินได้ / เห็ดพิษ)
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    
    if not os.path.isdir(category_path):
        continue  # ข้ามไฟล์ที่ไม่ใช่โฟลเดอร์

    images = os.listdir(category_path)
    
    # แบ่งข้อมูลเป็น train, val, test
    train_files, test_files = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # ฟังก์ชันช่วยคัดลอกไฟล์
    def move_files(file_list, target_folder):
        os.makedirs(target_folder, exist_ok=True)
        for file in file_list:
            src = os.path.join(category_path, file)
            dst = os.path.join(target_folder, file)
            shutil.copy2(src, dst)

    # คัดลอกไฟล์ไปยังโฟลเดอร์ Train, Val, Test
    move_files(train_files, os.path.join(output_folder, "train", category))
    move_files(val_files, os.path.join(output_folder, "val", category))
    move_files(test_files, os.path.join(output_folder, "test", category))

print("การแบ่งข้อมูลเสร็จสิ้น!")
