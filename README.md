# AI-Powered Financial Risk Prediction System

Live Demo: https://ai-powered-financial-risk-prediction-system-bfh2sqmagtpe6pr6hx.streamlit.app/

---

##  Overview

This project is an end-to-end **AI-powered financial risk prediction system** designed to evaluate loan applications and predict approval probability using machine learning.

It simulates a real-world **FinTech decision-making system** by combining data analysis, feature engineering, and predictive modeling with an interactive web interface.

Financial institutions rely heavily on accurate risk assessment to minimize loan defaults and improve decision-making efficiency, and machine learning models have proven highly effective for this task. :contentReference[oaicite:0]{index=0}

---

##  Key Features

-  End-to-End ML Pipeline (EDA → Preprocessing → Training → Deployment)
-  Exploratory Data Analysis (EDA)
-  Feature Engineering:
  - Loan-to-Income Ratio
  - EMI Calculation
-  Machine Learning Models:
  - Logistic Regression (Baseline)
  - Random Forest (Optimized)
-  Class Imbalance Handling
-  Model Evaluation:
  - Accuracy
  - Precision / Recall / F1-score
  - Confusion Matrix
  - Cross Validation
-  Interactive Streamlit Web App
-  Real-time Prediction + Business Insights

---

##  Problem Statement

Loan approval decisions are critical in the financial industry. Incorrect approvals can lead to defaults, while incorrect rejections reduce business opportunities.

This system aims to:

- Predict whether a loan should be approved or rejected
- Identify key risk factors affecting decisions
- Provide actionable insights for decision-making

---

##  Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib  
- **Frontend:** Streamlit  
- **Deployment:** Streamlit Cloud  

---

##  Project Architecture

```

financial-risk-project/
│
├── data/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
├── models/
│   └── model.pkl
├── app/
│   └── app.py
├── plots/
├── README.md

```

---

##  Workflow

```

Data Collection
↓
EDA & Insights
↓
Feature Engineering
↓
Model Training
↓
Evaluation & Validation
↓
Deployment (Streamlit)

````

---

##  Model Performance

###  Logistic Regression
- Balanced performance
- Good generalization
- Handles real-world scenarios better

###  Random Forest
- High accuracy
- Captures complex patterns
- Requires overfitting control

Machine learning models such as Random Forest and Logistic Regression are widely used in financial risk prediction due to their ability to capture patterns in borrower data and improve decision accuracy. :contentReference[oaicite:1]{index=1}

---

##  Key Insights

-  Credit Score is the strongest predictor of loan approval  
-  High Loan-to-Income Ratio increases risk  
-  Low Income applicants are more likely to be rejected  
-  Employment status impacts approval probability  

---

##  How to Run Locally

### 1️. Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd financial-risk-project
````

### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️. Train Model

```bash
python src/train.py
```

### 4️. Run App

```bash
streamlit run app/app.py
```

---

##  Some Screenshots
<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/4f5f37e7-e30d-4d8e-bd39-0ea80f68d054" />

* Approved Application
  
<img width="1917" height="1016" alt="image" src="https://github.com/user-attachments/assets/82e0b40a-55bb-4b19-90a4-dff871260f08" />

* Rejected Application

<img width="1918" height="1016" alt="image" src="https://github.com/user-attachments/assets/5e9eccc4-148d-4187-8028-c14f036fb465" />






---

##  Future Improvements

*  Add XGBoost / LightGBM models
*  Add SHAP explainability
*  Connect real-time financial APIs
*  Add deep learning models
*  Deploy on cloud (AWS / GCP)

---

##  Resume Highlight

Developed an AI-powered financial risk prediction system using machine learning and Streamlit, enabling real-time loan approval prediction with feature engineering, model evaluation, and business insights.

---

##  Contributing

Contributions are welcome! Feel free to fork the repo and submit pull requests.

---

##  License

This project is licensed under the MIT License.

---

##  Author

**Devraj Choudhary**

* GitHub: [https://github.com/CodingWithDevraj](https://github.com/CodingWithDevraj)
* LinkedIn: [https://www.linkedin.com/in/devraj-choudhary-3889412bb/](https://www.linkedin.com/in/devraj-choudhary-3889412bb/)

---

⭐ If you found this project useful, consider giving it a star!

```

