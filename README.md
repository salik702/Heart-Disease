<div align="center">

<!-- Animated Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=HEART%20PREDICTOR&fontSize=62&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Cardiac%20Risk%20Intelligence%20Engine&descAlignY=60&descSize=20&descColor=ffffff" width="100%"/>

<!-- Typing Animation -->
<a href="https://heart-disease-salik-knn.streamlit.app/">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=22&duration=3000&pause=800&color=F87171&center=true&vCenter=true&multiline=false&repeat=true&width=750&height=50&lines=❤️+Real-Time+Cardiac+Risk+Assessment;🧠+Powered+by+K-Nearest+Neighbors+(KNN);📊+86%25+Accuracy+·+Vital+Health+Metrics" alt="Typing SVG" />
</a>

<br/>

<!-- Badges Row -->
<p>
  <img src="https://img.shields.io/badge/STATUS-LIVE-brightgreen?style=for-the-badge&logo=statuspage&logoColor=white&labelColor=0d0d0d" />
  <img src="https://img.shields.io/badge/ACCURACY-86%25-f87171?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=0d0d0d" />
  <img src="https://img.shields.io/badge/ALGORITHM-KNN-ef4444?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=0d0d0d" />
  <img src="https://img.shields.io/badge/LICENSE-OPEN_SOURCE-fb923c?style=for-the-badge&logo=opensourceinitiative&logoColor=white&labelColor=0d0d0d" />
</p>

<p>
  <a href="https://heart-disease-salik-knn.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20LAUNCH%20APP-LIVE%20DEMO-f87171?style=for-the-badge&labelColor=1a0d0d" alt="Live Demo" />
  </a>
  &nbsp;
  <a href="https://salikahmad.vercel.app/">
    <img src="https://img.shields.io/badge/🌐%20PORTFOLIO-SALIK%20AHMAD-38bdf8?style=for-the-badge&labelColor=0d0d1a" alt="Portfolio" />
  </a>
</p>

</div>

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ❤️ What is Heart Disease Predictor?

<div align="center">

> *"Early detection saves lives. Let the data be your stethoscope."*

</div>

The **Heart Disease Predictor** is a next-generation **Cardiac Risk Intelligence Engine** — a production-grade Streamlit application powered by a K-Nearest Neighbors (KNN) classifier. It estimates the likelihood of heart disease by analyzing key patient vitals and health metrics in real time.

Whether you're a **healthcare professional** screening patients, a **medical researcher** studying cardiac risk factors, or a **health enthusiast** monitoring your own wellness — this tool delivers **instant, data-driven cardiac risk assessments** wrapped in a clean, intuitive interface.

<br/>

<!-- Risk Level Table -->
<div align="center">

| Risk Level | Likelihood | Classification |
|:----------:|:----------:|:--------------:|
| 🟢 Low Risk | 0% – 30% | `HEALTHY PROFILE` |
| 🟡 Moderate Risk | 31% – 65% | `MONITOR CLOSELY` |
| 🔴 High Risk | 66% – 100% | `SEEK MEDICAL ADVICE` |

</div>

---

## 🖼️ Preview

<div align="center">
  <img width="944" height="364" alt="Heart Disease Predictor UI Preview" src="https://github.com/user-attachments/assets/a0f8d601-b6bf-4a28-b166-c3975d39f30f" />
  <br/>
  <sub><i>✨ Clean, intuitive UI — where machine learning meets clinical decision support</i></sub>
</div>

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🚀 Key Features

<table>
<tr>
<td width="50%">

### 🔮 Real-Time Risk Prediction
Leverages a **K-Nearest Neighbors (KNN)** classifier to estimate heart disease probability instantly, comparing patient vitals against the closest matching profiles in the training dataset.

</td>
<td width="50%">

### 📊 Comprehensive Health Metrics
Accepts a full suite of patient inputs — **Age, Blood Pressure, Cholesterol, Resting ECG, Max Heart Rate, Chest Pain Type**, and more — for a holistic cardiac profile.

</td>
</tr>
<tr>
<td width="50%">

### 🧠 Instant KNN Inference
No waiting — predictions are generated in real time as inputs are provided, giving healthcare professionals and patients immediate feedback on cardiac risk levels.

</td>
<td width="50%">

### 🖥️ Streamlit-Powered Interface
A simple, distraction-free UI designed for both medical professionals and non-technical users — no setup friction, just input and predict.

</td>
</tr>
</table>

---

## ⚙️ How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│          HEART PREDICTOR — KNN PIPELINE                          │
└──────────────────────────────────────────────────────────────────┘

  🩺 PATIENT INPUT
   └─ Age · Blood Pressure · Cholesterol · ECG · Heart Rate · etc.
        │
        ▼
  🔧 PREPROCESSING
   └─ Feature Scaling (StandardScaler) · Encoding Categorical Variables
        │
        ▼
  📐 DISTANCE CALCULATION
   └─ Euclidean distance computed against all training samples
        │
        ▼
  🧠 K-NEAREST NEIGHBORS (KNN)
   └─ Top K closest patients retrieved → majority class vote
        │
        ▼
  ❤️ RISK OUTPUT
   └─ Heart Disease: YES / NO + Probability Score + Risk Category
```

---

## 📈 Model Performance

<div align="center">

| Metric | Value |
|:-------|:-----:|
| 🧠 Algorithm | K-Nearest Neighbors (KNN) |
| 🎯 Accuracy | **86%** |
| 📋 Input Features | Age, BP, Cholesterol, ECG, Heart Rate & more |
| ⚡ Inference | Real-time (< 100ms) |
| 🏥 Use Case | Binary Classification (Disease / No Disease) |

</div>

<div align="center">

```
Key Input Feature Impact on Prediction

Chest Pain Type      ████████████████████░  ~91%
Max Heart Rate       ███████████████████░░  ~87%
ST Depression        ██████████████████░░░  ~84%
Number of Vessels    █████████████████░░░░  ~80%
Cholesterol Level    ████████████████░░░░░  ~75%
Resting Blood Pres.  █████████████░░░░░░░░  ~63%
```

</div>

---

## 🛠️ Tech Stack

<div align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/JupyterLab-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />

</div>

---

## 📦 Local Setup

```bash
# 1️⃣ Clone the repository
git clone <your-repo-url>
cd heart-disease-predictor

# 2️⃣ Create & activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Launch 🚀
streamlit run app.py
```

> **Tip:** The app will auto-open at `http://localhost:8501` — no browser config needed.

---

## 🤝 Contributing

All contributions are welcome — whether it's improving model accuracy, adding new health metrics, or enhancing the UI.

```
1. Fork the repository
2. Create your feature branch  →  git checkout -b feature/CholesterolTrendAnalysis
3. Commit your changes         →  git commit -m "Add: Cholesterol trend visualization"
4. Push to branch              →  git push origin feature/CholesterolTrendAnalysis
5. Open a Pull Request         →  describe your changes clearly
```

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📧 Connect

<div align="center">

<h3>Built with obsession by <b>Salik Ahmad</b> ❤️</h3>

<p>
  <a href="https://salikahmad.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Website-salikahmad.vercel.app-f87171?style=for-the-badge&logo=vercel&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://www.linkedin.com/in/salik-ahmad-programmer/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Salik%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://www.kaggle.com/salikahmad702" target="_blank">
    <img src="https://img.shields.io/badge/Kaggle-salikahmad702-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white&labelColor=0d0d0d" />
  </a>
</p>

<br/>

<!-- Animated Footer Typing -->
<a href="https://salikahmad.vercel.app/">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=14&duration=4000&pause=1000&color=F87171&center=true&vCenter=true&width=700&lines=AI%2FML+Engineer+·+Healthcare+AI+·+UI+Craftsman;Copyright+©+2026+Salik+Ahmad.+All+rights+reserved.;Built+with+❤️+and+a+mission+to+save+lives+through+data." alt="Footer Typing" />
</a>

<br/><br/>

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

</div>
