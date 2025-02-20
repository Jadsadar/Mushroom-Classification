import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ฟังก์ชันเบลอภาพ
def blur_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

# ฟังก์ชันหมุนภาพ
def rotate_image(img, angle, zoom_factor=1.7):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # แปลงเป็น PIL
    rotated = pil_img.rotate(angle, expand=True)  # หมุน
    
    # คำนวณขนาดภาพใหม่หลังซูม
    width, height = rotated.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Crop ตรงกลาง (ซูมเข้า)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped = rotated.crop((left, top, right, bottom))
    # Resize กลับไปเป็นขนาดเดิม
    zoomed = cropped.resize((width, height), Image.LANCZOS)
    
    cvtColor = cv2.cvtColor(np.array(zoomed), cv2.COLOR_RGB2BGR)
    return cv2.resize(cvtColor, (512, 512))
    

# ฟังก์ชันเพิ่ม Noise (Gaussian Noise)
def add_noise(img, mean=10, sigma=0):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img

# กำหนดโฟลเดอร์ข้อมูลหลัก
data_folder = "data"

# อ่านโฟลเดอร์ใน data_folder (แต่ละโฟลเดอร์คือชนิดของเห็ด)almond_mushroom
for class_folder in os.listdir(data_folder):
    class_path = os.path.join(data_folder, class_folder)
    if not os.path.isdir(class_path):  # ข้ามถ้าไม่ใช่โฟลเดอร์
        continue

# class_path = os.path.join(data_folder, "almond_mushroom")
# if not os.path.isdir(class_path):  # ข้ามถ้าไม่ใช่โฟลเดอร์
#     print('not Folder')

    # อ่านจำนวนภาพในโฟลเดอร์
    all_images = [f for f in os.listdir(class_path) if f.endswith(".png")]
    all = len(all_images)  # จำนวนภาพทั้งหมด
    count = 0

        # วนลูปอ่านภาพ
    for i in range(all):
        img_path = os.path.join(class_path, f"{i}.png")
        if not os.path.exists(img_path):  # ข้ามถ้าภาพไม่มี
            continue
        
        img = cv2.imread(img_path)

        # บันทึกต้นฉบับเป็นชื่อใหม่
        img_n = img
        cv2.imwrite(os.path.join(class_path, f"00{count}.png"), img_n)
        count += 1

        # เพิ่มเบลอและบันทึก
        blurred = blur_image(img)
        cv2.imwrite(os.path.join(class_path, f"00{count}.png"), blurred)
        count += 1

        # หมุนซ้ายและบันทึก
        img_l = rotate_image(img, -15)
        img_l_noisy = add_noise(img_l)
        cv2.imwrite(os.path.join(class_path, f"00{count}.png"), img_l_noisy)
        count += 1

        # หมุนขวาและบันทึก
        img_r = rotate_image(img, 15)
        img_r_noisy = add_noise(img_r)
        cv2.imwrite(os.path.join(class_path, f"00{count}.png"), img_r_noisy)
        count += 1

        try:
            os.remove(img_path)
            print(f" ลบ {img_path} สำเร็จ")
        except Exception as e:
            print(f" ลบ {img_path} ไม่สำเร็จ: {e}")

print("Data Augmentation done")