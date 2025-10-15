
# 🧠 Crowd Density Detection – Explainable AI

> An AI-powered system that detects people in images, estimates crowd density levels, and provides explainable visual feedback for public safety and crowd management.

---

## 📸 Overview

This project combines **Deep Learning (YOLOv8)** and **TensorFlow** to:
- Detect people in images.
- Estimate **crowd density** (safe, moderate, or high risk).
- Visualize detections with bounding boxes and confidence scores.
- Train a **custom regression model** to predict crowd size using transfer learning (InceptionResNetV2).
- Provide an interactive **Streamlit web app** for real-time detection.

---

## 🚀 Features

### 🧩 Streamlit App
- Upload an image and get **real-time crowd detection**.
- Automatically classifies the scene as:
  - 🟩 **Not Crowded (Safe)**
  - 🟧 **Moderate (Caution)**
  - 🟥 **Highly Crowded (Alert)**
- Displays bounding boxes with **confidence scores**.
- Provides **Explainable AI** visualization of model predictions.

### 🧠 Model Training
- Uses **YOLOv8** for people detection.
- Trains a regression model using **TensorFlow InceptionResNetV2** to predict person counts.
- Includes data augmentation, early stopping, and fine-tuning.

---

## 🗂️ Project Structure

```

crowd_project/  
├─ frames/ # Folder with frame images (seq_000001.jpg, etc.)  
├─ images.npy # Preprocessed dataset (optional)  
├─ labels.csv # Labels file (columns: id, count)  
├─ app.py # Streamlit Explainable AI app  
├─ crowd_management.py # YOLOv8 + TensorFlow training pipeline  
├─ best_model.h5 # Trained TensorFlow model (auto-saved)  
├─ requirements.txt # List of dependencies  
└─ README.md # Project documentation

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/crowd-density-detection.git
cd crowd-density-detection
````

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
        

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit tensorflow tensorflow-hub ultralytics numpy pandas matplotlib pillow tqdm scikit-learn
```

---

## 🧪 Running the Streamlit App

Once dependencies are installed, launch the Explainable AI dashboard:

```bash
streamlit run app.py
```

open the URL displayed in the terminal  `http://localhost:8501`.

Upload an image and view:

- 👥 People count
    
- 🤖 Model confidence
    
- ⚠️ Risk level
    
- 📊 Bounding boxes and explanations
    

---

## 🧬 Training the Model (Optional)

To retrain or fine-tune the detection model:

1. Ensure `images.npy`, `labels.csv`, and `frames/` exist in your project folder.
    
2. Run the training script:
    

```bash
python crowd_management.py
```

This will:

- Load and preprocess data.
    
- Train a deep regression model to predict crowd counts.
    
- Save the best model as `best_model.h5`.
    

Training progress and metrics will appear in the terminal and TensorBoard (`logs/` directory).

---

## 📈 Results & Metrics

- **MAE (Mean Absolute Error):** Average difference between actual and predicted crowd count.
    
- **MSE (Mean Squared Error):** Penalizes large errors more strongly.
    
- Fine-tuning improves accuracy for large and diverse crowd scenes.
    


---

## 💡 Applications

- 🏙️ **Smart City Monitoring**
    
- 🧍‍♂️ **Public Safety Analysis**
    
- 🎟️ **Event Management**
    
- 🚌 **Transportation Hubs**
    
- 🏫 **Campus or Mall Security**
    

---

## 🧠 Technologies Used

|Category|Tools / Frameworks|
|---|---|
|**Object Detection**|YOLOv8 (Ultralytics)|
|**Deep Learning**|TensorFlow, Keras, TensorFlow Hub|
|**Visualization**|Streamlit, Matplotlib|
|**Data Handling**|NumPy, Pandas|
|**Explainable AI**|Bounding box visualizations, confidence scores|

---

## 📌 Future Improvements

- 🔄 Integrate **real-time webcam feed** detection.
    
- 📹 Support **video crowd tracking**.
    
- 🌐 Deploy app to **Streamlit Cloud** or **Hugging Face Spaces**.
    
- 📊 Add **Grad-CAM** or **heatmap explainability** for deeper insight.
    

---

## 👨‍💻 Author & Credits

**Developed by:**  
🚀 _NEXT GENTHINKERS (2025)_  

> “Empowering safety through vision-based intelligence.”

---

## 🪪 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute for educational or research purposes.

---

## 🌟 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
    
- [TensorFlow Hub Models](https://tfhub.dev/)
    
- [Streamlit](https://streamlit.io/)
    

