🚀 Duality AI – Space Station Safety Object Detection Challenge

Live Demo: 🔗 Streamlit App

📌 Overview

This repository contains our solution for the “Safety Object Detection #2” Hackathon, organized by Duality AI.

The challenge was to build a robust object detection model capable of identifying seven critical safety objects inside a simulated space station.

The dataset was generated using Duality AI’s Falcon digital twin platform, providing diverse synthetic images with realistic variations in lighting, occlusion, and camera perspectives.

Our solution uses YOLOv8 for object detection and includes a Streamlit web app for real-time inference, deployed on Streamlit Cloud.

🎯 Problem Statement

Detect the following safety-critical objects:

🟦 OxygenTank

🟦 NitrogenTank

🟩 FirstAidBox

🔴 FireAlarm

⚡ SafetySwitchPanel

☎️ EmergencyPhone

🔥 FireExtinguisher

🛠️ Technical Workflow
1️⃣ Environment Setup
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/My Drive/Ms_hack

!unzip Hackathon2_scripts.zip
!unzip Hackathon2_test1.zip
!unzip hackathon2_train_1.zip

2️⃣ Dependency Installation

We ensured a stable training environment using condacolab:

!pip install -q condacolab
import condacolab
condacolab.install()

!mamba install ultralytics opencv-contrib-python streamlit -y

3️⃣ Model Training (YOLOv8)

Base model: yolov8s.pt

Hyperparameters:

Epochs = 100

Mosaic = 1.0

Patience = 100

python train.py --model yolov8s.pt --epochs 100 --img 640

4️⃣ Performance Monitoring

We monitored:

mAP@0.5 (Mean Average Precision)

Loss curves (Box, Class, DFL)

Precision, Recall, F1 trends

5️⃣ Streamlit App

We developed an interactive Streamlit app for real-time safety object detection.

🔗 Live Deployment: Streamlit App

▶️ Run Locally

Clone the repo:

git clone https://github.com/<your-repo>/dualityai-safety-detection.git
cd dualityai-safety-detection


Install dependencies:

pip install -r requirements.txt


Start the app:

streamlit run app.py


Open browser → http://localhost:8501

📊 Results

mAP@0.5: 0.993 (synthetic validation set)

Precision & Recall: Very high, confirming robust detection

⚠️ Note: Performance on real-world images may differ → domain adaptation required

📦 Deliverables

✅ YOLOv8 trained weights (best.pt)

✅ Training & inference scripts (train.py, predict.py)

✅ Streamlit Web App (app.py) + deployment

✅ Performance Report (PDF/DOCX)

✅ README with setup instructions

⭐ Bonus: Falcon-based retraining pipeline proposal

🏆 Judging Criteria Alignment

Model Performance (mAP@0.5): 80 pts

Report Clarity & Reproducibility: 20 pts

Bonus (App + Falcon Use-case): Extra credit

📌 Future Improvements

Domain adaptation for real-world generalization

Fine-tuning with real images

Extend to multi-modal monitoring (Vision + IoT sensors)

Continuous digital twin retraining pipeline with Falcon

🙌 Contributors

MD Waseem Ahmed

Vantala Saisree

MD Rukhnuddin

MD Faizan

Mujihad Ahmed

T Sai Nikhil

🔥 This project shows how synthetic data + YOLOv8 + Streamlit can be harnessed for critical safety monitoring in space missions.
