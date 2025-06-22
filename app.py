import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Customer Churn SHAP Dashboard", layout="wide")

st.markdown("""
# Customer Churn - SHAP Explainability  
Use this interactive dashboard to explore how customer features affect churn predictions.
""")

# Sidebar options
st.sidebar.header("Settings")
show_waterfall = st.sidebar.checkbox("Show Waterfall Plot", value=False)
show_decision = st.sidebar.checkbox("Show Decision Plot", value=False)
show_interaction = st.sidebar.checkbox("Show Interaction Plot", value=False)
max_display = st.sidebar.slider("Max Features in SHAP Plots", 5, 20, 10)
compare_mode = st.sidebar.checkbox("Compare Two Customers", value=False)

# Upload dataset
st.sidebar.markdown("### Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Churn" not in df.columns:
        st.error("Uploaded file must contain a 'Churn' column.")
        st.stop()
    st.success("Custom dataset loaded.")
else:
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Filter by churn label
churn_filter = st.sidebar.radio("Filter by Churn Label", options=["All", "Churned", "Not Churned"])
if churn_filter == "Churned":
    df = df[df["Churn"] == "Yes"]
elif churn_filter == "Not Churned":
    df = df[df["Churn"] == "No"]

# Preprocess data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop("Churn", axis=1)
y = df["Churn"]
X = pd.get_dummies(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Check for binary class balance
if y.nunique() < 2:
    st.error("The filtered dataset does not contain both churned and not churned customers. Please select 'All' in the sidebar.")
    st.stop()

# Train-test split and fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy:.2%}")
st.sidebar.metric("ROC AUC", f"{auc:.2f}")

# SHAP values
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
expected_value = explainer.expected_value

# Select customer(s)
index1 = st.slider("Select customer index 1", 0, X_test.shape[0] - 1, 0)
if compare_mode:
    index2 = st.slider("Select customer index 2", 0, X_test.shape[0] - 1, 1)

# Prediction display
proba1 = model.predict_proba([X_test.iloc[index1]])[0][1]
st.metric(label=f"Customer {index1} Churn Probability", value=f"{proba1:.2%}")
if compare_mode:
    proba2 = model.predict_proba([X_test.iloc[index2]])[0][1]
    st.metric(label=f"Customer {index2} Churn Probability", value=f"{proba2:.2%}")

# SHAP explanation tabs
tabs = st.tabs(["Force", "Summary", "Static Scatter", "Interactive Scatter"] +
               (["Waterfall"] if show_waterfall else []) +
               (["Decision"] if show_decision else []) +
               (["Interaction"] if show_interaction else []))

figures = []

with tabs[0]:
    st.subheader("SHAP Force Plot")
    shap.initjs()
    force_plot_html_path = "force_plot.html"
    shap.save_html(force_plot_html_path, shap.force_plot(expected_value, shap_values.values[index1], X_test.iloc[index1]))
    with open(force_plot_html_path, 'r') as f:
        html_string = f.read()
        st.components.v1.html(html_string, height=400)

with tabs[1]:
    st.subheader("SHAP Summary Plot")
    fig_summary = plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    st.pyplot(fig_summary)
    figures.append(fig_summary)

with tabs[2]:
    st.subheader("SHAP Scatter Plot: MonthlyCharges")
    feature = "MonthlyCharges"
    feature_index = X_test.columns.get_loc(feature)
    fig_static, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_test[feature],
        shap_values.values[:, feature_index],
        c=shap_values.values[:, feature_index],
        cmap="viridis"
    )
    ax.set_xlabel(f"{feature} value")
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"SHAP Scatter Plot for {feature}")
    fig_static.colorbar(scatter, ax=ax, label="SHAP Value")
    st.pyplot(fig_static)
    figures.append(fig_static)

with tabs[3]:
    st.subheader("Interactive SHAP Scatter Plot")
    selected_feature = st.selectbox("Choose a feature", X_test.columns)
    selected_index = X_test.columns.get_loc(selected_feature)
    fig_plotly = px.scatter(
        x=X_test[selected_feature],
        y=shap_values.values[:, selected_index],
        color=shap_values.values[:, selected_index],
        labels={
            "x": f"{selected_feature} value",
            "y": f"SHAP value for {selected_feature}",
            "color": "SHAP value"
        },
        title=f"SHAP Scatter Plot (Hover Enabled) - {selected_feature}",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

if show_waterfall:
    st.subheader("SHAP Waterfall Plot")
    fig_waterfall, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[index1], max_display=max_display, show=False)
    st.pyplot(fig_waterfall)
    figures.append(fig_waterfall)

if show_decision:
    st.subheader("SHAP Decision Plot")
    fig_decision = plt.figure(figsize=(10, 6))
    shap.decision_plot(expected_value, shap_values.values[index1], X_test.iloc[index1], show=False)
    st.pyplot(fig_decision)
    figures.append(fig_decision)

if show_interaction:
    with tabs[6 if show_waterfall and show_decision else 5 if show_waterfall or show_decision else 4]:
        st.subheader("SHAP Interaction Plot")
        X_test_reset = X_test.reset_index(drop=True)
        shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test_reset)
        fig_inter = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_interaction_values, X_test_reset, plot_type="dot", show=False)
        st.pyplot(fig_inter)
        figures.append(fig_inter)

# Save to PDF
os.makedirs("plots", exist_ok=True)
pdf_path = "plots/shap_visualizations.pdf"
with PdfPages(pdf_path) as pdf:
    for fig in figures:
        try:
            pdf.savefig(fig, bbox_inches='tight')
        except Exception as e:
            st.warning(f"Could not save one of the plots to PDF: {e}")

with open(pdf_path, "rb") as f:
    st.download_button(
        label="📄 Download All Visualizations as PDF",
        data=f,
        file_name="shap_visualizations.pdf",
        mime="application/pdf"
    )

# Tooltip / Help section
st.info("""
**What do SHAP plots show?**  
- **Force plot** explains the contribution of each feature to a single prediction.  
- **Summary plot** gives an overview of feature importance and distribution.  
- **Waterfall** breaks down how each feature pushed the prediction higher or lower.  
- **Decision plot** shows the decision path in the model.  
- **Interaction plot** shows pairwise feature interactions.
""")
