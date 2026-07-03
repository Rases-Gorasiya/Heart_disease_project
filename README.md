# 🫀 Heart Disease Prediction App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end Machine Learning project that predicts the likelihood of heart disease based on clinical parameters — deployed as an interactive Streamlit web application.**

[📊 Live Demo](#how-to-run) · [📁 Project Structure](#project-structure) · [🚀 Features](#features)

</div>

---

## 📌 Introduction

Cardiovascular disease is one of the leading causes of death globally. Early and accurate detection can save lives. This project builds a **Machine Learning classification system** trained on real-world clinical data to predict whether a patient is at risk of heart disease.

The model is deployed as an **interactive Streamlit web app**, where users can input their clinical parameters and instantly receive a prediction. The entire pipeline — from data cleaning and EDA to model selection and deployment — is fully implemented in Python.

---

## ✨ Features

- 🔍 **Exploratory Data Analysis (EDA)** — Distribution plots, count plots, heatmaps, and box plots
- 🧹 **Data Cleaning** — Handles zero-value anomalies in `Cholesterol` and `RestingBP`
- ⚙️ **Feature Engineering** — One-hot encoding of categorical variables
- 📏 **Feature Scaling** — StandardScaler applied to numerical features
- 🤖 **Multi-Model Comparison** — 5 ML models evaluated and compared
- 🏆 **Best Model Selection** — Logistic Regression selected with ~86% accuracy
- 💾 **Model Serialization** — Model, scaler, and column names saved with `joblib`
- 🌐 **Streamlit Web App** — Real-time, interactive prediction interface
- 📦 **Reproducible Pipeline** — All artifacts saved for consistent predictions

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Source** | [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| **Rows** | 918 |
| **Columns** | 12 |
| **Target** | `HeartDisease` (0 = No Disease, 1 = Disease) |
| **Task** | Binary Classification |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Age of the patient (years) |
| `Sex` | Categorical | M = Male, F = Female |
| `ChestPainType` | Categorical | TA, ATA, NAP, ASY |
| `RestingBP` | Numerical | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numerical | Serum cholesterol (mg/dl) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No) |
| `RestingECG` | Categorical | Normal, ST, LVH |
| `MaxHR` | Numerical | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (Y/N) |
| `Oldpeak` | Numerical | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease` | Binary | **Target variable** (0 or 1) |

---

## 🔬 Machine Learning Workflow

```
Raw Data (heart.csv)
       │
       ▼
   Data Loading & EDA
       │
       ▼
   Data Cleaning
   (Replace zero Cholesterol & RestingBP with column means)
       │
       ▼
   Feature Engineering
   (pd.get_dummies → One-Hot Encoding of categorical columns)
       │
       ▼
   Feature Scaling
   (StandardScaler on numerical columns)
       │
       ▼
   Train-Test Split (80/20, random_state=42)
       │
       ▼
   Train & Evaluate 5 ML Models
       │
       ▼
   Select Best Model (Logistic Regression)
       │
       ▼
   Save Model, Scaler & Columns (joblib)
       │
       ▼
   Streamlit Web App (app.py)
```

---

## 🤖 Models Used

| Model | Accuracy | F1 Score |
|---|---|---|
| **Logistic Regression** ✅ | ~86% | ~87% |
| Decision Tree Classifier | ~80% | ~82% |
| Gaussian Naive Bayes | ~84% | ~86% |
| K-Nearest Neighbors | ~83% | ~85% |
| Support Vector Machine | ~84% | ~85% |

> ✅ **Logistic Regression** was selected as the best-performing model based on Accuracy and F1 Score.

---

## 🛠️ Technologies

| Category | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Streamlit |
| ML Library | scikit-learn |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |

---

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/Rases-Gorasiya/Heart_disease_project.git
cd Heart_disease_project
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your default browser at:
```
http://localhost:8501
```

### Using the App

1. Adjust the **Age** and **Maximum Heart Rate** sliders
2. Select your **Sex**, **Chest Pain Type**, and other clinical options
3. Enter your **Resting BP**, **Cholesterol**, and **Oldpeak** values
4. Click the **"Predict"** button
5. The app will instantly display whether the model predicts heart disease or not

---

## 📁 Project Structure

```
Heart_disease_project/
│
├── app.py                        # Streamlit web application
├── heart_project.py              # ML pipeline: EDA, preprocessing, training
├── heart.csv                     # Raw dataset
│
├── best_heart_disease_model.pkl  # Trained Logistic Regression model
├── scaler.pkl                    # Fitted StandardScaler
├── model_columns.pkl             # Feature column names for alignment
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🖼️ Screenshots

> 📸 *Screenshots will be added after the app is deployed.*

| App Home Page | Prediction Result |
|---|---|
| *(Screenshot placeholder)* | *(Screenshot placeholder)* |

---

## 🔮 Future Improvements

- [ ] 🌐 Deploy to **Streamlit Cloud** or **Hugging Face Spaces** for public access
- [ ] 📊 Add an **interactive EDA dashboard** within the Streamlit app
- [ ] 🧪 Implement **cross-validation** and **hyperparameter tuning** (GridSearchCV)
- [ ] 🌳 Explore **ensemble methods** (Random Forest, XGBoost, LightGBM)
- [ ] 📈 Add **SHAP/LIME explainability** to show feature importances
- [ ] 📱 Make the UI fully **mobile-responsive**
- [ ] 🔄 Add a **batch prediction** feature via CSV file upload
- [ ] 🛡️ Add **input validation** and better error handling in the UI

---

## ⚠️ Disclaimer

> **This application is intended for educational and research purposes only.**
> It is NOT a medical diagnostic tool and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare provider for any health concerns.

---

## 👤 Author

**Rases Gorasiya**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Rases-Gorasiya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rases-gorasiya-57aba0370/?skipRedirect=true)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](gorasiyarashes@gmail.com)


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If you found this project useful, please consider giving it a star!** ⭐

Made with ❤️ and Python

</div>
