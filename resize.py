import os
import cv2

data_folder = "data"
data_resize_folder = "data_resize"
os.makedirs(data_resize_folder, exist_ok=True)  # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี

resize_size = (128, 128)  # กำหนดขนาดใหม่

# อ่านโฟลเดอร์ใน data_folder (แต่ละโฟลเดอร์คือประเภทของเห็ด)
for class_folder in os.listdir(data_folder):
    class_path = os.path.join(data_folder, class_folder)
    class_resize_path = os.path.join(data_resize_folder, class_folder)
    os.makedirs(class_resize_path, exist_ok=True)  # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    
    if not os.path.isdir(class_path):  # ข้ามถ้าไม่ใช่โฟลเดอร์
        continue
    
    # วนลูปอ่านทุกไฟล์ภาพในโฟลเดอร์
    for file_name in os.listdir(class_path):
        img_path = os.path.join(class_path, file_name)
        save_path = os.path.join(class_resize_path, file_name)
        
        # ตรวจสอบว่าเป็นไฟล์ภาพ
        if not (file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg")):
            continue
        
        # อ่านภาพ
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ ไม่สามารถอ่านไฟล์: {img_path}")
            continue
        
        # Resize ภาพ
        img_resized = cv2.resize(img, resize_size)
        
        # บันทึกภาพใน data_resize
        cv2.imwrite(save_path, img_resized)
        print(f"✅ Resized and saved: {save_path}")

print("🎉 Resize เสร็จสิ้น!")
