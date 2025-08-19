# 🛡️ Weapon Detection Using YOLOv11

This project implements a **Weapon Detection Model** capable of identifying **pistols and knives** in real-time environments using deep learning.  
The system is based on the **YOLOv11 object detection architecture** and is trained on a publicly available dataset to achieve robust performance under varying lighting and environmental conditions.  

---

## 🔍 Project Overview
The model is trained using a labeled dataset from **[Kaggle](https://www.kaggle.com/datasets/iqmansingh/guns-knives-object-detection)** and fine-tuned to detect pistols and knives.  
After training on high-performance GPUs, the system demonstrates an average accuracy of approximately **90% (mAP)**.  

📓 **Model Training Code:** Available on Kaggle — [View Notebook](https://www.kaggle.com/code/overwatch2003/weapon-detection-model-yolo11/notebook?scriptVersionId=256756529)
The objective of the project is to provide a real-time weapon detection solution that can be integrated into security systems, surveillance applications, or law enforcement technology.  

---

## 🛠️ Tech Stack & Key Highlights
- ✅ **Dataset Preparation** — Cleaning, annotation, and structuring of thousands of weapon images  
- ✅ **Model Training** — Fine-tuned YOLOv11 on an **NVIDIA Tesla P100 GPU**  
- ✅ **Data Augmentation** — Applied image preprocessing and augmentation to improve robustness  
- ✅ **Optimization** — Hyperparameter tuning for improved accuracy and faster convergence  
- ✅ **Deployment Pipeline** — The model currently runs locally, with planned deployment using **Flask**  
- ✅ **Process Flow** — `Dataset → Model Training → Evaluation → Deployment`  

---

## 🚀 Key Learnings
- Handling real-world challenges in object detection and image classification  
- Techniques for dataset curation, annotation, and quality control  
- Practical methods for improving model accuracy and reducing false positives  
- Workflow design for training, evaluation, and deployment in a collaborative environment  

---

## 📦 Installation & Usage

### 1. Clone the Repository:
git clone <your-repo-link>.git
cd weapon-detection-yolov11


### 2. Create a Virtual Environment:
python -m venv myvenv
source myvenv/bin/activate # On Mac/Linux
myvenv\Scripts\activate # On Windows


### 3. Install Dependencies:
pip install -r requirements.txt


### 4. Run Inference:
python detect.py --source path/to/video_or_image --weights runs/train/best.pt


---

## 📊 Results
- **Model Accuracy:** ~90% mAP  
- **Objects Detected:** Pistols, Knives  
- **Output:** Detected objects are highlighted with bounding boxes in video or image streams  

---

## 📎 Deployment
- **GitHub Repo:** [Placeholder Link]  
- **Online Deployment:** *Planned using Flask for web-based inference*  

---

## 👨‍💻 Contributors
- **[Your Name]** — Data collection, preprocessing, model training, performance evaluation  
- **[Shivam Kushwaha](https://github.com/ShivamKushwaha20)** — Deployment pipeline and integration  

---

## 🙌 Acknowledgements
- **YOLOv11 community** for the object detection framework  
- **Kaggle dataset providers** for making annotated data available 

---
