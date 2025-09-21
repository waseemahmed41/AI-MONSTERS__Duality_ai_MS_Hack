# 🚀 Duality AI – Space Station Safety Object Detection Challenge

## 📌 Overview

This repository contains our solution for the **“Safety Object Detection #2” Hackathon**, organized by **Duality AI**.

The challenge focuses on developing a robust **object detection model** capable of identifying **seven critical safety objects** inside a **simulated space station**. The dataset was generated using **Duality AI’s Falcon digital twin platform**, offering diverse synthetic images with realistic conditions such as varying lighting, occlusion, and camera perspectives.

By leveraging **YOLOv8** and synthetic training data, this project demonstrates how digital twins can accelerate AI development in environments that are otherwise costly or dangerous to access.

---

## 🎯 Problem Statement

The objective is to detect the following safety objects within the simulated environment:

* 🟦 **OxygenTank**
* 🟦 **NitrogenTank**
* 🟩 **FirstAidBox**
* 🔴 **FireAlarm**
* ⚡ **SafetySwitchPanel**
* ☎️ **EmergencyPhone**
* 🔥 **FireExtinguisher**

---

## 🛠️ Technical Workflow

### 1️⃣ Environment Setup

* Mount Google Drive
* Navigate to the project directory
* Unzip datasets and provided scripts

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/My Drive/Ms_hack

!unzip Hackathon2_scripts.zip
!unzip Hackathon2_test1.zip
!unzip hackathon2_train_1.zip
```

---

### 2️⃣ Dependency Installation

We use **condacolab** to ensure a compatible environment for training:

```bash
!pip install -q condacolab
import condacolab
condacolab.install()

!mamba install ultralytics opencv-contrib-python -y
```

---

### 3️⃣ Model Training (YOLOv8)

* Base model: **yolov8s.pt**
* Hyperparameters:

  * Epochs = `100`
  * Mosaic = `1.0`
  * Patience = `100`

Training script:

```bash
python train.py --model yolov8s.pt --epochs 100 --img 640
```



### 4️⃣ Performance Monitoring

During training, monitor:

* **mAP\@0.5** (Mean Average Precision)
* **Loss curves** (Box loss, Classification loss, DFL loss)
* **Precision, Recall, and F1-Score trends**


## 📊 Results

* **mAP\@0.5:** `0.993` (on synthetic validation set)
* **Recall & F1-Score:** High values, confirming robust detection on synthetic data
* ⚠️ **Note:** Synthetic performance does not guarantee real-world generalization → further testing on real images is required.



## 📦 Deliverables

* ✅ Trained YOLO model weights (`best.pt`)
* ✅ Training & inference scripts (`train.py`, `predict.py`)
* ✅ Performance Evaluation Report (PDF/DOCX)
* ✅ README.md with setup & usage instructions
* ⭐ Bonus: Use-case application ideas for maintaining/updating models using Falcon


## 🏆 Judging Criteria

* **Model Performance (mAP\@0.5):** 80 points
* **Performance Report clarity:** 20 points
* **Bonus (Use-case app with Falcon):** Extra points



## 📌 Future Improvements

* Enhance **real-world generalization** with domain adaptation
* Fine-tune with **real images** where available
* Extend to **multi-modal safety monitoring** (vision + IoT sensors)
* Continuous updates via **Falcon digital twin retraining pipeline**


## 🙌 Contributors 

* **MD Waseem Ahmed**
* **Vantala Saisree**
* **MD Rukhnuddin**
* **MD Faizan**
* **Mujihad Ahmed**
* **T Sai Nikhil**


🔥 With this solution, we show how **synthetic data + YOLOv8** can be harnessed for **critical safety applications in space missions**.

