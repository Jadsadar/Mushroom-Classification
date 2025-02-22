import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ตั้งค่าพารามิเตอร์
IMAGE_SIZE = (512, 512)  # ขนาดภาพที่ใช้
BATCH_SIZE = 128  # ขนาด Batch
EPOCHS = 20  # จำนวนรอบที่เทรน
DATASET_PATH = "data"  # โฟลเดอร์เก็บข้อมูล (แต่ละโฟลเดอร์คือแต่ละประเภทของเห็ด)

# ✨ โหลดข้อมูลและทำ Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,  # ทำให้ค่า pixel อยู่ในช่วง 0-1
    horizontal_flip=True,  # กลับภาพแนวนอน
    validation_split=0.3  # แบ่งข้อมูลเป็น train 80% และ validation 20%
)

# โหลดข้อมูลสำหรับเทรน
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # ใช้เป็นชุด Training
)

# โหลดข้อมูลสำหรับ Validation
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # ใช้เป็นชุด Validation
)

# ✨ สร้างโมเดล CNN
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # ลด Overfitting
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output Layer (ตามจำนวนเห็ดที่มี)
])

# ✨ คอมไพล์โมเดล
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✨ แสดงโครงสร้างโมเดล
model.summary()

# ✨ เทรนโมเดล
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ✨ บันทึกโมเดลที่เทรนแล้ว
model.save("mushroom_cnn_model.h5")

print("✅ โมเดลเทรนเสร็จแล้ว และถูกบันทึกเป็น 'mushroom_cnn_model.h5' 🎉")
