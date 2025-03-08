import streamlit as st
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
import numpy as np

class MushroomModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=22, out_features=50)
        self.layer_2 = nn.Linear(in_features=50, out_features=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        return self.layer_2(self.relu(self.layer_1(x)))


# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="เห็ดอิหยัง", layout="wide")

# ใช้ CSS ปรับแต่ง Sidebar ให้เป็นการ์ด
st.markdown(
    """
    <style>
        .sidebar-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
            border: 2px solid transparent;
        }
        .selected-card {
            background-color: #007bff !important;
            color: white !important;
            border: 2px solid #0056b3;
            font-weight: bold;
        }
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "selected_option" not in st.session_state:
    st.session_state.selected_option = "default"

st.sidebar.title("🍄 Menu 🔍🍄")

menu_options = ["วิเคราะห์เห็ดด้วยภาพ", "วิเคราะห์เห็ดแบบละเอียด"]

for option in menu_options:
    if st.sidebar.button(option, use_container_width=True):
        st.session_state.selected_option = option

st.title("เห็ดอิหยัง 🍄")

if st.session_state.selected_option == "default":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            ### 🧠 วิเคราะห์เห็ดด้วย AI
            ยินดีต้อนรับสู่แอปพลิเคชัน "เห็ดอิหยัง"  
            แอปนี้ช่วยให้คุณสามารถวิเคราะห์ชนิดของเห็ดได้โดยใช้ **AI**  
            - 🔍 วิเคราะห์จากภาพถ่าย  
            - 📋 วิเคราะห์จากคุณสมบัติต่างๆ ของเห็ด  
              
            **กรุณาเลือกเมนูทางด้านซ้ายเพื่อเริ่มต้น**  
            """
        )

    with col2:
        st.image("mush_page.webp", caption="", use_container_width=True)

elif st.session_state.selected_option == "วิเคราะห์เห็ดด้วยภาพ":
    st.subheader("อัปโหลดภาพเห็ดของคุณ")
    uploaded_file = st.file_uploader("เลือกไฟล์", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="ภาพที่อัปโหลด", use_container_width=True)

        if st.button("🔍 วิเคราะห์"):
            model = YOLO("best.pt")
            image_size = 64
            
            # ปรับขนาดภาพ
            image_resized = cv2.resize(image, (image_size, image_size))

            # ทำการทำนาย
            results = model.predict(image_resized)

            # ตรวจสอบว่าผลลัพธ์มีข้อมูลหรือไม่
            if len(results) > 0:
                confidence = results[0].probs
                confidence_enable_mush = confidence.data[0] * 100  # แปลงเป็นเปอร์เซ็นต์
                confidence_poison_mush = confidence.data[1] * 100   # แปลงเป็นเปอร์เซ็นต์

                # แสดงผลลัพธ์
                if confidence_enable_mush > confidence_poison_mush:
                    st.success(f"✅ ผลลัพธ์: เห็ดชนิดนี้กินได้ (ความมั่นใจ: {confidence_enable_mush:.2f}%)")
                else:
                    st.error(f"❌ ผลลัพธ์: เห็ดชนิดนี้เป็นพิษ (ความมั่นใจ: {confidence_poison_mush:.2f}%)")
            else:
                st.error("❌ ไม่มีข้อมูลในการทำนาย")

elif st.session_state.selected_option == "วิเคราะห์เห็ดแบบละเอียด":
    st.subheader("ป้อนข้อมูลเห็ด")
    attributes = {
        "รูปร่างหมวกเห็ด": ["ระฆัง", "กรวย", "โค้งนูน", "แบน", "มียอดแหลม", "บุ๋มลึก"],
        "พื้นผิวหมวกเห็ด": ["เป็นเส้นใย", "มีร่อง", "เป็นเกล็ด", "เรียบ"],
        "สีหมวกเห็ด": ["น้ำตาล", "ครีม", "อบเชย", "เทา", "เขียว", "ชมพู", "ม่วง", "แดง", "ขาว", "เหลือง"],
        "มีรอยช้ำหรือไม่": ["มี", "ไม่มี"],
        "กลิ่น": ["อัลมอนด์", "โป๊ยกั๊ก", "กลิ่นไหม้", "กลิ่นปลา", "กลิ่นเหม็น", "กลิ่นอับ", "ไม่มีกลิ่น", "กลิ่นฉุน", "กลิ่นเผ็ด"],
        "การติดของครีบ": ["ติดก้าน", "ห้อยลง", "ไม่ติด", "มีรอยบาก"],
        "ระยะห่างของครีบ": ["ชิด", "แน่น", "ห่าง"],
        "ขนาดของครีบ": ["กว้าง", "แคบ"],
        "สีของครีบ": ["ดำ", "น้ำตาล", "ครีม", "ช็อกโกแลต", "เทา", "เขียว", "ส้ม", "ชมพู", "ม่วง", "แดง", "ขาว", "เหลือง"],
        "รูปทรงก้าน": ["ขยายขึ้น", "เรียวลง"],
        "รากของก้าน": ["หัวกลม", "กระบอก", "ถ้วย", "เท่ากัน", "มีเส้นใยราก", "ฝังรากลึก", "ไม่มีข้อมูล"],
        "พื้นผิวก้านเหนือวงแหวน": ["เป็นเส้นใย", "เป็นเกล็ด", "เป็นมันเงา", "เรียบ"],
        "พื้นผิวก้านใต้วงแหวน": ["เป็นเส้นใย", "เป็นเกล็ด", "เป็นมันเงา", "เรียบ"],
        "สีของก้านเหนือวงแหวน": ["น้ำตาล", "ครีม", "อบเชย", "เทา", "ส้ม", "ชมพู", "แดง", "ขาว", "เหลือง"],
        "สีของก้านใต้วงแหวน": ["น้ำตาล", "ครีม", "อบเชย", "เทา", "ส้ม", "ชมพู", "แดง", "ขาว", "เหลือง"],
        "ประเภทของวงแหวน": ["บางส่วน", "ทั้งหมด"],
        "สีของวงแหวน": ["น้ำตาล", "ส้ม", "ขาว", "เหลือง"],
        "จำนวนวงแหวน": ["ไม่มี", "หนึ่ง", "สอง"],
        "ลักษณะของวงแหวน": ["เป็นใยแมงมุม", "สลายตัวได้", "บานออก", "ใหญ่", "ไม่มี", "ห้อยลง", "เป็นปลอก", "เป็นโซน"],
        "สีของสปอร์": ["ดำ", "น้ำตาล", "ครีม", "ช็อกโกแลต", "เขียว", "ส้ม", "ม่วง", "ขาว", "เหลือง"],
        "ประชากร": ["หนาแน่น", "เป็นกลุ่ม", "จำนวนมาก", "กระจัดกระจาย", "หลายต้น", "ต้นเดียว"],
        "ที่อยู่อาศัย": ["หญ้า", "ใบไม้", "ทุ่งหญ้า", "ทางเดิน", "เขตเมือง", "กองขยะ", "ป่า"]
    }
    encoding_dict = {
        "รูปร่างหมวกเห็ด": {"ระฆัง": 0, "กรวย": 1, "โค้งนูน": 2, "แบน": 3, "มียอดแหลม": 4, "บุ๋มลึก": 5},
        "พื้นผิวหมวกเห็ด": {"เป็นเส้นใย": 0, "มีร่อง": 1, "เป็นเกล็ด": 2, "เรียบ": 3},
        "สีหมวกเห็ด": {"น้ำตาล": 0, "ครีม": 1, "อบเชย": 2, "เทา": 3, "เขียว": 4, "ชมพู": 5, "ม่วง": 6, "แดง": 7, "ขาว": 8, "เหลือง": 9},
        "มีรอยช้ำหรือไม่": {"มี": 1, "ไม่มี": 0},
        "กลิ่น": {"อัลมอนด์": 0, "โป๊ยกั๊ก": 1, "กลิ่นไหม้": 2, "กลิ่นปลา": 3, "กลิ่นเหม็น": 4, "กลิ่นอับ": 5, "ไม่มีกลิ่น": 6, "กลิ่นฉุน": 7, "กลิ่นเผ็ด": 8},
        "การติดของครีบ": {"ติดก้าน": 0, "ห้อยลง": 1, "ไม่ติด": 2, "มีรอยบาก": 3},
        "ระยะห่างของครีบ": {"ชิด": 0, "แน่น": 1, "ห่าง": 2},
        "ขนาดของครีบ": {"กว้าง": 0, "แคบ": 1},
        "สีของครีบ": {"ดำ": 0, "น้ำตาล": 1, "ครีม": 2, "ช็อกโกแลต": 3, "เทา": 4, "เขียว": 5, "ส้ม": 6, "ชมพู": 7, "ม่วง": 8, "แดง": 9, "ขาว": 10, "เหลือง": 11},
        "รูปทรงก้าน": {"ขยายขึ้น": 0, "เรียวลง": 1},
        "รากของก้าน": {"หัวกลม": 0, "กระบอก": 1, "ถ้วย": 2, "เท่ากัน": 3, "มีเส้นใยราก": 4, "ฝังรากลึก": 5, "ไม่มีข้อมูล": 6},
        "พื้นผิวก้านเหนือวงแหวน": {"เป็นเส้นใย": 0, "เป็นเกล็ด": 1, "เป็นมันเงา": 2, "เรียบ": 3},
        "พื้นผิวก้านใต้วงแหวน": {"เป็นเส้นใย": 0, "เป็นเกล็ด": 1, "เป็นมันเงา": 2, "เรียบ": 3},
        "สีของก้านเหนือวงแหวน": {"น้ำตาล": 0, "ครีม": 1, "อบเชย": 2, "เทา": 3, "ส้ม": 4, "ชมพู": 5, "แดง": 6, "ขาว": 7, "เหลือง": 8},
        "สีของก้านใต้วงแหวน": {"น้ำตาล": 0, "ครีม": 1, "อบเชย": 2, "เทา": 3, "ส้ม": 4, "ชมพู": 5, "แดง": 6, "ขาว": 7, "เหลือง": 8},
        "ประเภทของวงแหวน": {"บางส่วน": 0, "ทั้งหมด": 1},
        "สีของวงแหวน": {"น้ำตาล": 0, "ส้ม": 1, "ขาว": 2, "เหลือง": 3},
        "จำนวนวงแหวน": {"ไม่มี": 0, "หนึ่ง": 1, "สอง": 2},
        "ลักษณะของวงแหวน": {"เป็นใยแมงมุม": 0, "สลายตัวได้": 1, "บานออก": 2, "ใหญ่": 3, "ไม่มี": 4, "ห้อยลง": 5, "เป็นปลอก": 6, "เป็นโซน": 7},
        "สีของสปอร์": {"ดำ": 0, "น้ำตาล": 1, "ครีม": 2, "ช็อกโกแลต": 3, "เขียว": 4, "ส้ม": 5, "ม่วง": 6, "ขาว": 7, "เหลือง": 8},
        "ประชากร": {"หนาแน่น": 0, "เป็นกลุ่ม": 1, "จำนวนมาก": 2, "กระจัดกระจาย": 3, "หลายต้น": 4, "ต้นเดียว": 5},
        "ที่อยู่อาศัย": {"หญ้า": 0, "ใบไม้": 1, "ทุ่งหญ้า": 2, "ทางเดิน": 3, "เขตเมือง": 4, "กองขยะ": 5, "ป่า": 6}
    }

    user_input = {}
    for key, values in attributes.items():
        user_input[key] = st.selectbox(key, values)

    encoded_values = [encoding_dict[key][value] for key, value in user_input.items()]
    
    if len(encoded_values) == 22:
        X_input = torch.tensor([encoded_values], dtype=torch.float32)

        # โหลดโมเดล
        model = MushroomModelV0()  # ต้องแน่ใจว่ามีการสร้างโมเดลนี้
        model.load_state_dict(torch.load("ClassificationModel0.pth"))
        model.eval()  # ตั้งโมเดลให้อยู่ในโหมดประเมิน

        if st.button("🔍 วิเคราะห์"):
            with torch.no_grad():
                prediction = model(X_input)
                probability = torch.sigmoid(prediction)  # ค่าความน่าจะเป็นเป็น Tensor
                predicted_class = torch.round(probability).item()  # ใช้ torch.round และแปลงเป็น float

            confidence_percentage = probability.item() * 100  # แปลงเป็นเปอร์เซ็นต์

            if predicted_class == 1:
                st.success(f"✅ ผลลัพธ์: เห็ดชนิดนี้กินได้ (ความมั่นใจ: {confidence_percentage:.2f}%)")
            else:
                st.error(f"❌ ผลลัพธ์: เห็ดชนิดนี้เป็นพิษ (ความมั่นใจ: {confidence_percentage:.2f}%)")
    else:
        st.error("❌ ข้อมูลไม่ครบ 22 ฟีเจอร์ กรุณาตรวจสอบและกรอกข้อมูลให้ครบ.")

    