# Fraud Detection — Credit Card Transactions

This project builds a machine learning model to detect fraudulent credit card transactions using Python.  
It includes data loading, preprocessing, EDA, model training (Logistic Regression, Random Forest, XGBoost), and evaluation.

---

## 📌 Project Overview
- Developed an end-to-end fraud detection ML pipeline.
- Dataset used: Public Kaggle Credit Card Fraud dataset.
- Models implemented: Logistic Regression, Random Forest, XGBoost.
- Evaluation: Accuracy, Precision, Recall, F1 score, Confusion Matrix.
- Objective: Maximize Recall for the fraud class (important in fraud detection).

---

## 📂 Dataset
- Source: Kaggle  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- Total rows: ~284,807  
- Fraud cases: Less than 1%  
- Features: 30 (Time, V1–V28, Amount, Class)

`Class` → 0 = Genuine, 1 = Fraudulent

---

## 🛠️ Tech Stack Used
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Imbalanced-learn** (optional for SMOTE)
- **Jupyter Notebook**

---

## 🚀 Project Workflow

### **1️⃣ Data Loading**
Loaded the CSV file and viewed basic structure (head(), info(), describe()).

### **2️⃣ Exploratory Data Analysis**
- Checked fraud vs non-fraud imbalance  
- Visualized distributions  
- Created a correlation heatmap  
- Plotted Amount by fraud status  

### **3️⃣ Preprocessing**
- Train-test split  
- StandardScaler applied to `Amount`  
- Optionally applied SMOTE  

### **4️⃣ Model Building**
Models used:
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### **5️⃣ Model Evaluation**
Used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC-AUC score  

Confusion matrix and classification report were plotted.

---

## 📊 Example Code Snippets

### Train-test split & scaling
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test['Amount'] = scaler.transform(X_test[['Amount']])
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred)
📈 Results

Accuracy: 99%

Precision: 92%

Recall (Fraud class): 80%

F1-score: 86
