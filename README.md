# Portuguese Bank Marketing Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview

The Portuguese Bank Marketing Prediction project uses Machine Learning to predict whether a customer will subscribe to a bank term deposit based on demographic information, banking details, previous campaign outcomes, and economic indicators.

The goal is to help banks identify potential customers and improve marketing campaign strategies.

## Live Application

**Try the deployed application:**

[Portuguese Bank Marketing Prediction – Live Demo](https://portuguese-bank-marketing-prediction-deployment.streamlit.app/)

## Project Objectives

- Analyze customer and marketing campaign data.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Build and evaluate Machine Learning models.
- Identify important features influencing predictions.
- Select the best-performing model.
- Predict customer term deposit subscriptions.

## Dataset Information

**Dataset:** Portuguese Bank Marketing Dataset

| Description | Value |
|---|---:|
| Original records | 41,188 |
| Records after cleaning | 41,176 |
| Original columns | 21 |
| Duplicate records removed | 12 |
| Target variable | `y` |

**Target Variable:**

- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Algorithms

Six classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Linear Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 87.02% | 44.44% | 60.78% | **51.34%** | 0.8091 |
| Decision Tree | 84.76% | 38.67% | 60.13% | 47.07% | 0.7688 |
| Linear SVM | 83.11% | 36.10% | 64.76% | 46.36% | 0.8001 |
| Logistic Regression | 83.03% | 35.91% | 64.55% | 46.15% | 0.8002 |
| KNN | 89.55% | 56.61% | 30.93% | 40.00% | 0.7404 |
| Gradient Boosting | **89.95%** | **65.06%** | 23.28% | 34.29% | 0.8093 |

## Final Model

**Selected Model: Random Forest Classifier**

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 87.02% |
| Precision | 44.44% |
| Recall | 60.78% |
| F1 Score | 51.34% |
| ROC-AUC | 0.8091 |

The Random Forest model achieved the highest F1-score among the six evaluated models, making it the selected model for this project.

## Feature Importance

The top influential features identified by Random Forest include:

1. `euribor3m`
2. `nr.employed`
3. `emp.var.rate`
4. `age`
5. `cons.conf.idx`
6. `campaign`
7. `pdays`
8. `cons.price.idx`
9. `poutcome_success`
10. `contact_cellular`

## Project Structure

```text
ProtugeseBank/
│
├── Data/
│   ├── bank.csv
│   └── bank-names.txt
│
├── ProtugeseBank.ipynb
├── app.py
├── requirements.txt
├── random_forest_bank_model.pkl
├── bank_encoder.pkl
├── bank_scaler.pkl
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Gopinath-P88/Portuguese-Bank-Marketing-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Portuguese-Bank-Marketing-Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

### 5. Open the Jupyter Notebook

```bash
jupyter notebook
```

Open `ProtugeseBank.ipynb` to explore the complete analysis and model development process.

## Model Files

The following trained model and preprocessing files are included:

- `random_forest_bank_model.pkl` – Trained Random Forest model.
- `bank_encoder.pkl` – Fitted categorical encoder.
- `bank_scaler.pkl` – Fitted numerical scaler.

## Business Applications

- Identify potential term deposit subscribers.
- Improve customer targeting.
- Support marketing campaign planning.
- Reduce unnecessary customer contacts.
- Assist data-driven business decisions.

## Future Improvements

- Hyperparameter tuning.
- SMOTE and other class imbalance techniques.
- XGBoost model comparison.
- SHAP-based model explainability.
- Model monitoring and performance tracking.

## Author

**Gopinath P**

- GitHub: [Gopinath-P88](https://github.com/Gopinath-P88)
- LinkedIn: [Gopinath P](https://www.linkedin.com/in/gopinath-p21/)

## Project Repository

[Portuguese Bank Marketing Prediction – GitHub](https://github.com/Gopinath-P88/Portuguese-Bank-Marketing-Prediction)

---

*This project was developed as part of a Data Science and Machine Learning learning project.*