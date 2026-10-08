import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

# Load model
model = joblib.load('model.pkl')

st.title("Bank Customer Churn Predictor")
st.write("Enter customer details to predict churn risk and see what's driving it.")

# Input form
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", 0.0, 300000.0, 60000.0)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Build input row matching training columns
input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': 1 if has_cr_card == "Yes" else 0,
    'IsActiveMember': 1 if is_active == "Yes" else 0,
    'EstimatedSalary': salary,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Churn Probability", f"{prob*100:.1f}%")

    if prob > 0.5:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")

        # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.subheader("Why this prediction?")
    fig, ax = plt.subplots()
    shap_explanation = shap.Explanation(
        values=shap_values[0, :, 1],
        base_values=explainer.expected_value[1],
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist()
    )
    shap.plots.waterfall(shap_explanation, show=False)
    st.pyplot(fig)
