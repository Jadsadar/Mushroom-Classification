import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

def augment_images(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg', 'webp', 'jfif'))]
    print('ภาพที่เจอ ', len(image_files))

    global_counter = 0  # ใช้ตัวนับเพื่อให้ไฟล์ไม่ซ้ำกัน
    
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))  # ปรับขนาดภาพเป็น 256x256
        img_pil = Image.open(img_path).convert("RGB").resize((256, 256))
        base_name = os.path.splitext(img_name)[0]
        
        augmentations = []
        
        # 1. Flip ซ้ายไปขวา
        augmentations.append(cv2.flip(img, 1))
        
        # 2. Flip ซ้ายไปขวาแล้วกลับหัว
        augmentations.append(cv2.flip(img, -1))
        
        # 3. หมุนภาพเอียงไปทางซ้าย (-30 องศา)
        rows, cols = img.shape[:2]
        M_left = cv2.getRotationMatrix2D((cols/2, rows/2), -30, 1)
        augmentations.append(cv2.warpAffine(img, M_left, (cols, rows)))
        
        # 4. หมุนภาพเอียงไปทางขวา (+30 องศา)
        M_right = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
        augmentations.append(cv2.warpAffine(img, M_right, (cols, rows)))
        
        # 5. Gaussian Blur (เพิ่มความเบลอมากขึ้น)
        augmentations.append(cv2.GaussianBlur(img, (9,9), 2))
        
        # 6. เพิ่ม Blur Filter แบบ PIL
        blurred = img_pil.filter(ImageFilter.GaussianBlur(3))
        blurred = blurred.resize((256, 256))
        augmentations.append(cv2.cvtColor(np.array(blurred), cv2.COLOR_RGB2BGR))
        
        # 7. ปรับ Brightness (ลดลง)
        augmentations.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-30))
        
        # 8. ปรับ Contrast (เพิ่มขึ้น)
        augmentations.append(cv2.convertScaleAbs(img, alpha=1.8, beta=0))
        
        # 9. Crop และ Resize
        h, w = img.shape[:2]
        cropped = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        augmentations.append(cv2.resize(cropped, (256, 256)))
        
        # 10. Sharpen Filter
        sharpened = img_pil.filter(ImageFilter.SHARPEN)
        sharpened = sharpened.resize((256, 256))
        augmentations.append(cv2.cvtColor(np.array(sharpened), cv2.COLOR_RGB2BGR))
        
        # บันทึกภาพต้นฉบับ
        output_path = os.path.join(output_folder, f"p_{global_counter:05d}.png")
        cv2.imwrite(output_path, img)
        global_counter += 1  # เพิ่มตัวนับ
        
        for aug_img in augmentations:
            output_path = os.path.join(output_folder, f"p_{global_counter}.png")
            cv2.imwrite(output_path, aug_img)
            global_counter += 1  # เพิ่มตัวนับไม่ให้ซ้ำกัน
            print(global_counter)

    print('end')

# เรียกใช้งาน
input_folder = "resource/raw_pois_mush"  # เปลี่ยนเป็นโฟลเดอร์ของภาพต้นฉบับ
output_folder = "resource/poison_mush"  # เปลี่ยนเป็นโฟลเดอร์สำหรับบันทึกภาพที่ทำ AUG
augment_images(input_folder, output_folder)
