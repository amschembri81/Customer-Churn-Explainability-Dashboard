# 📊 Customer Churn Explainability Dashboard

An interactive Streamlit app that visualizes SHAP values to explain predictions of customer churn. Built using XGBoost, SHAP, and the Telco Customer Churn dataset. This tool is ideal for business analysts and data scientists who want to understand **why** a model predicts churn for specific customers.

---

## 🚀 Features

- Upload and explore your own CSV dataset
- Train an XGBoost model with preprocessing
- Visualize model predictions using:
  - SHAP Force Plot
  - SHAP Summary Plot
  - SHAP Waterfall Plot
  - SHAP Decision Plot
  - SHAP Interaction Plot
  - Static and Interactive SHAP Scatter Plots
- Export all visualizations to a PDF

---

## 🧰 Tech Stack

- Python
- Streamlit
- SHAP
- XGBoost
- Scikit-learn
- Pandas
- Matplotlib / Plotly

---

## 🗂️ File Structure

Project 6/

├── app.py                          # Streamlit app

├── requirements.txt                # Python dependencies

├── README.md                       # Project README (this file)

├── data/

│ └── Telco-Customer-Churn.csv      # Sample dataset

├── plots/

│ └── shap_visualizations.pdf       # Exported visuals (generated)

├── notebooks/

│ └── churn_analysis.ipynb          # EDA and model notebook

├── model.pkl                       # (optional) Saved model (currently empty)

├── scaler.pkl                      # (optional) Saved scaler (currently empty)

├── churn_app.py                    # (optional, currently unused)

└── .venv/                          # Python virtual environment


---

## 📦 Dataset

The app uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), included under `data/`.

---

## 📒 Notebooks

The `notebooks/churn_analysis.ipynb` includes:

- Exploratory Data Analysis (EDA)
- Preprocessing
- Model Training
- SHAP Value Generation

---

## 💡 SHAP Visualizations

The app uses [SHAP](https://github.com/slundberg/shap) to explain each customer’s churn prediction. You can interpret:

- 🔍 Which features push the prediction toward churn or retention
- 📈 Overall feature importance across the population
- 🔁 Interactions between pairs of features

---

## 🧪 Installation and Usage

### 1. Clone the Repo and Set Up Environment

```bash
git clone https://github.com/your-username/customer-churn-explainability-dashboard.git
cd customer-churn-explainability-dashboard
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the App

To start the application, run the following command:

```bash
streamlit run app.py
```

Once started, the app will automatically open in your default browser at:  http://localhost:8501

## 📄 Exporting Visuals
To export all SHAP plots:

Navigate to the application in your browser.

Click the "📄 Download All Visualizations as PDF" button.

The SHAP plots will be saved as a PDF in the following directory: plots/shap_visualizations.pdf
You can then download or access this file directly.

## 🛠️ To Do
 Add support for uploading a pre-trained model

 Allow user to select features for SHAP summary

 Deploy to Streamlit Community Cloud

## 💡 Why This Project?
Understanding why a customer is predicted to churn is just as important as knowing that they will. This project is a bridge between black-box models and human-centered interpretability.

Use cases:

Model debugging and validation

Customer insight discovery

Stakeholder-ready reporting

Feature importance exploration
