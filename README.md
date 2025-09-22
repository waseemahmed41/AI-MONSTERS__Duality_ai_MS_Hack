  <h1>🚀 Duality AI – Space Station Safety Object Detection Challenge</h1>
  <p><strong>Live Demo:</strong> 🔗 
    <a href="https://dualityai-spacestationdetection.streamlit.app/" target="_blank">
      Streamlit App
    </a>
  </p>
<p><strong>Dataset :</strong> 🔗 
    <a href="https://falcon.duality.ai/secure/documentation/7-class-hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=hyderabad" target="_blank">
      Falcon_Link
    </a>
  </p>
<h4>🎥 Demo Video</h4>
 <a href="https://res.cloudinary.com/dwxxznitz/video/upload/v1758529308/DualityAi_js5woa.mp4" target="_blank">    </a>
[![Watch the video](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://res.cloudinary.com/dwxxznitz/video/upload/v1758529308/DualityAi_js5woa.mp4)


  <hr>

  <h2>📌 Overview</h2>
  <p>
    This repository contains our solution for the <strong>“Safety Object Detection #2” Hackathon</strong>, 
    organized by <strong>Duality AI</strong>.
  </p>
  <p>
    The challenge required building a <strong>robust object detection system</strong> to identify 
    <strong>seven critical safety objects</strong> inside a <strong>simulated space station</strong>.
  </p>
  <p>
    The dataset was generated using <strong>Duality AI’s Falcon digital twin platform</strong>, 
    providing highly realistic synthetic images with variations in 
    <em>lighting, occlusion, and camera perspectives</em>.
  </p>
  <p>
    Our solution leverages <strong>YOLOv8</strong> for detection and an 
    <strong>interactive Streamlit app</strong> for real-time inference, 
    deployed on <strong>Streamlit Cloud</strong>.
  </p>

  <hr>

  <h2>🎯 Problem Statement</h2>
  <p>The model detects the following safety-critical objects:</p>
  <ul>
    <li>🟦 OxygenTank</li>
    <li>🟦 NitrogenTank</li>
    <li>🟩 FirstAidBox</li>
    <li>🔴 FireAlarm</li>
    <li>⚡ SafetySwitchPanel</li>
    <li>☎️ EmergencyPhone</li>
    <li>🔥 FireExtinguisher</li>
  </ul>

  <hr>

  <h2>🛠️ Technical Workflow</h2>

  <h3>1️⃣ Environment Setup</h3>
  <pre><code>from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/My Drive/Ms_hack

!unzip Hackathon2_scripts.zip
!unzip Hackathon2_test1.zip
!unzip hackathon2_train_1.zip
</code></pre>

  <h3>2️⃣ Dependency Installation</h3>
  <pre><code>!pip install -q condacolab
import condacolab
condacolab.install()

!mamba install ultralytics opencv-contrib-python streamlit -y
</code></pre>

  <h3>3️⃣ Model Training (YOLOv8)</h3>
  <ul>
    <li><strong>Base model:</strong> yolov8s.pt</li>
    <li><strong>Hyperparameters:</strong>
      <ul>
        <li>Epochs = 100</li>
        <li>Mosaic = 1.0</li>
        <li>Patience = 100</li>
      </ul>
    </li>
  </ul>
  <pre><code>python train.py --model yolov8s.pt --epochs 100 --img 640
</code></pre>

  <h3>4️⃣ Performance Monitoring</h3>
  <ul>
    <li>📈 mAP@0.5 (Mean Average Precision)</li>
    <li>📉 Loss curves (Box, Class, DFL)</li>
    <li>🔄 Precision, Recall, and F1-score trends</li>
  </ul>

  <h3>5️⃣ Streamlit Application</h3>
  <p>
    We built an <strong>interactive Streamlit app</strong> for real-time safety object detection.  
    Users can upload images and instantly see detected safety objects.
  </p>
  <p>
    🌐 <strong>Deployment:</strong> 
    <a href="https://dualityai-spacestationdetection.streamlit.app/" target="_blank">
      Streamlit App
    </a>
  </p>

  <h4>▶️ Run Locally</h4>
  <ol>
    <li>
      Clone the repo:
      <pre><code>git clone https://github.com/&lt;your-repo&gt;/dualityai-safety-detection.git
cd dualityai-safety-detection</code></pre>
    </li>
    <li>
      Install dependencies:
      <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>
      Run the app:
      <pre><code>streamlit run app.py</code></pre>
    </li>
    <li>Open in browser: <code>http://localhost:8501</code></li>
  </ol>

  <hr>

  <h2>📊 Results</h2>
  <ul>
    <li><strong>mAP@0.5:</strong> 0.993 (synthetic validation set)</li>
    <li><strong>Precision & Recall:</strong> Very high</li>
    <li><strong>Note:</strong> Real-world generalization requires further adaptation</li>
  </ul>

  <hr>

  <h2>📦 Deliverables</h2>
  <ul>
    <li>✅ YOLOv8 trained weights (best.pt)</li>
    <li>✅ Training & inference scripts (train.py, predict.py)</li>
    <li>✅ Streamlit Web App (app.py) + deployment</li>
    <li>✅ Performance Report (PDF/DOCX)</li>
    <li>✅ README with setup instructions</li>
    <li>⭐ Falcon-based retraining pipeline proposal</li>
  </ul>

  <hr>

  <hr>

  <h2>📌 Future Improvements</h2>
  <ul>
    <li>🌍 Domain adaptation for real-world deployment</li>
    <li>🖼️ Fine-tuning with real images</li>
    <li>📡 Multi-modal monitoring (Vision + IoT sensors)</li>
    <li>🔁 Continuous retraining pipeline using Falcon digital twin</li>
  </ul>

  <hr>

  <h2>🙌 Contributors</h2>
  <ul>
    <li>MD Waseem Ahmed</li>
    <li>Vantala Saisree</li>
    <li>MD Rukhnuddin</li>
    <li>MD Faizan</li>
    <li>Mujihad Ahmed</li>
    <li>T Sai Nikhil</li>
  </ul>

  <hr>

  <p><strong>🔥 This project shows how synthetic data + YOLOv8 + Streamlit can be harnessed 
    for mission-critical safety monitoring in space exploration.</strong></p>
