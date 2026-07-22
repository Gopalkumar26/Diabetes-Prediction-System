# 🩺 Diabetes Prediction System Using Machine Learning

A Machine Learning web application that predicts whether a person is diabetic or non-diabetic based on health parameters. The project is built using the PIMA Indians Diabetes Dataset and deployed with Streamlit.

## 🚀 Live Demo

🔗 https://diabetes-prediction-system-e5jf3nx9nbamlzyvcnef2v.streamlit.app

---

## 📌 Features

- Predicts diabetes using Machine Learning
- User-friendly Streamlit web application
- Real-time prediction
- Displays prediction probability
- Shows risk level
- Provides health recommendations
- Downloadable prediction report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📂 Dataset

- **Dataset:** PIMA Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Records:** 768
- **Features:** 8
- **Target Variable:**
  - 0 → Non-Diabetic
  - 1 → Diabetic

---

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Scaling using StandardScaler
5. Train-Test Split (80:20)
6. Logistic Regression Model Training
7. Model Evaluation
8. Model Saving
9. Streamlit Web Application
10. Deployment

---

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | ~71% |
| Precision | ~60% |
| Recall | ~51% |
| F1-Score | ~56% |
| ROC-AUC | ~82% |

---

## 📁 Project Structure

```
Diabetes-Prediction-System/
│
├── app.py
├── diabetes.csv
├── diabetes_prediction.ipynb
├── model_joblib.pkl
├── scaler.pkl
├── feature_names.pkl
├── requirements.txt
├── README.md
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/Gopalkumar26/Diabetes-Prediction-System.git
```

Move into the project folder

```bash
cd Diabetes-Prediction-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Screenshots

Add screenshots of:

- Home Page
- User Input Form
- Prediction Result
- Download Report

---

## 🔮 Future Improvements

- Random Forest and XGBoost
- Neural Networks
- Larger healthcare datasets
- Mobile application
- Hospital database integration
- Explainable AI (SHAP/LIME)

---

## 👨‍💻 Author

**Gopal Kumar**

B.Tech Computer Science and Engineering

Lovely Professional University

LinkedIn: *(Add your LinkedIn profile)*

GitHub: https://github.com/Gopalkumar26

---

## ⭐ If you found this project useful, please give it a star!
