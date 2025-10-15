import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Crowd Density Detection - Explainable AI",
    page_icon="🧠",
    layout="wide"
)

# ===================== STYLING =====================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    color: #f5f6fa;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    font-weight: 600;
}
h1 {
    color: #00f5d4 !important;
}
h2 {
    color: #9b5de5 !important;
}
.stButton>button {
    background: linear-gradient(45deg, #00f5d4, #9b5de5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #9b5de5, #00f5d4);
    transform: scale(1.05);
}
.sidebar .sidebar-content {
    background: #1b1b2f;
    color: white;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 10px;
}
.upload-box {
    border: 2px dashed #00f5d4;
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}
.result-box {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}
.footer {
    text-align: center;
    color: #9b9b9b;
    margin-top: 40px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR INFO =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4727/4727470.png", width=120)
    st.markdown("## 🔍 About the Project")
    st.write("""
    This project uses **Deep Learning (YOLOv8)** for detecting
    people and estimating **crowd density** in a scene.
    It classifies the image as:
    - 🟩 **Not Crowded** → Safe  
    - 🟧 **Moderate** → Manageable  
    - 🟥 **Highly Crowded** → Risk Alert  
    """)
    st.markdown("### 🧠 Explainable AI Insights")
    st.write("""
    - Each detected person is identified via bounding boxes.  
    - The **confidence score** (0-1) shows model certainty.  
    - The system estimates **risk level** based on density.  
    - It helps in **smart city**, **public safety**, and **event monitoring** applications.
    """)
    st.markdown("---")
    st.write("👨‍💻 *Developed by NEXT GENTHINKERS (2025)*")

# ===================== MAIN HEADER =====================
st.markdown("<h1>🧠 Crowd Density Detection with Explainable AI</h1>", unsafe_allow_html=True)
st.markdown("<h4>Empowering safety through vision-based intelligence</h4>", unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

with st.spinner("🚀 Loading deep learning model..."):
    detector = load_model()
st.success("✅ Model loaded successfully!")

# ===================== FUNCTIONS =====================
def detect_objects(image):
    tensor = tf.image.decode_jpeg(tf.io.encode_jpeg(np.array(image)), channels=3)[tf.newaxis, ...]
    return detector(tensor)

def count_persons(results, threshold=0.3):
    classes = results["detection_classes"].numpy()[0]
    scores = results["detection_scores"].numpy()[0]
    valid = (classes == 1) & (scores > threshold)
    return np.sum(valid), np.mean(scores[valid]) if np.any(valid) else 0

def draw_boxes(image, results, threshold=0.3):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    boxes = results["detection_boxes"].numpy()[0]
    classes = results["detection_classes"].numpy()[0].astype(int)
    scores = results["detection_scores"].numpy()[0]
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            draw.rectangle(
                [(xmin*w, ymin*h), (xmax*w, ymax*h)],
                outline="#00f5d4",
                width=3
            )
            draw.text((xmin*w, ymin*h), f"{scores[i]:.2f}", fill="#00f5d4")
    return image

# ===================== UPLOAD AREA =====================
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    with st.spinner("🤖 Detecting people and analyzing scene..."):
        results = detect_objects(image)
        count, confidence = count_persons(results, threshold=0.3)
        output_img = draw_boxes(image.copy(), results, threshold=0.3)

    # ===================== CROWD LEVEL =====================
    if count == 0:
        label, color = "No People Detected 🟦", "#0077b6"
        risk = "Low Risk"
    elif count < 6:
        label, color = "Not Crowded 🟩", "#06d6a0"
        risk = "Safe Zone"
    elif count < 12:
        label, color = "Moderately Crowded 🟧", "#ffd166"
        risk = "Caution Zone"
    else:
        label, color = "Highly Crowded 🟥", "#ef476f"
        risk = "High Risk"

    # ===================== RESULT DISPLAY =====================
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.image(output_img, caption=f"Detected People: {count}", use_column_width=True)
    st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>👥 Count</h3><h2>{count}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>🤖 Avg Confidence</h3><h2>{confidence*100:.1f}%</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>⚠️ Risk Level</h3><h2>{risk}</h2></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===================== EXPLANATION SECTION =====================
    st.markdown("### 🧩 Model Explanation")
    st.write("""
    The model analyzes pixel-level patterns using **SSD MobileNet V2**, 
    a convolutional neural network pre-trained on the **COCO dataset**.
    Each bounding box indicates a detected person with a confidence score.
    - Confidence > 0.6 → Strong detection  
    - Confidence < 0.3 → Possibly background noise  
    The final crowd classification is determined by the total number 
    of high-confidence detections.
    """)
else:
    st.info("👆 Upload an image to start detection.")

# ===================== FOOTER =====================
st.markdown("<div class='footer'>© 2025 Sujan | AI Crowd Management Project</div>", unsafe_allow_html=True)
