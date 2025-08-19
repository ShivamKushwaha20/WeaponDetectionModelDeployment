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
```git clone <your-repo-link>.git```<br/>
```cd weapon-detection-yolov11```


### 2. Create a Virtual Environment:
```python -m venv myvenv```<br/>
```source myvenv/bin/activate # On Mac/Linux```<br/>
```myvenv\Scripts\activate # On Windows```


### 3. Install Dependencies:
```pip install -r requirements.txt```


### 4. Run the Flask App:
```python app.py```


---

## 📊 Results
- **Model Accuracy:** ~90% mAP  
- **Objects Detected:** Pistols, Knives  
- **Outputs:** Detected objects are highlighted with bounding boxes in video or image streams
  
![--------_------_jpg rf 4eb0868f6cd41827c921043ddfa37ff9](https://github.com/user-attachments/assets/24687b56-2e8f-468d-a68e-f097b5df189d)
![445_jpg rf 2e04379013684f454abbc00564910fcc](https://github.com/user-attachments/assets/a233e2a2-6f4b-47d5-980b-ab4663821c7e)
![armas--1118-_jpg rf 822747298e61e608e1a7c545effeade3](https://github.com/user-attachments/assets/9192c091-e17c-4eb7-932a-295522c1e980)
![armas--1172-_jpg rf cf59adba26d811d154f10f64ea781427](https://github.com/user-attachments/assets/e34c012a-c53b-4b73-8be5-e47e87e6e6d2)


---

## 📂 Code Repository
- **GitHub Repo (Inference Code & Model Training Results):** [https://github.com/ShivamKushwaha20/Weapon_detection_system]

---

## 👨‍💻 Contributors
- **[Vaibhav Sharma](https://github.com/torq125)** — Data collection, preprocessing, model training, performance evaluation  
- **[Shivam Kushwaha](https://github.com/ShivamKushwaha20)** — Deployment pipeline and integration  

---

## 🙌 Acknowledgements
- **YOLOv11 community** for the object detection framework  
- **Kaggle dataset providers** for making annotated data available 

---
