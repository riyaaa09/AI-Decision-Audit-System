import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load column names
with open("data_columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.set_page_config(page_title="AI Decision Audit System", layout="wide")

st.title("🧠 AI-Driven Decision Audit System")
st.write("Analyze, explain, and simulate business decisions before execution.")

st.sidebar.header("📊 Enter Decision Inputs")

price_change = st.sidebar.slider("Price Change (%)", -20, 30, 0)
demand = st.sidebar.selectbox("Demand Level", ["Low", "Medium", "High"])
competitor = st.sidebar.selectbox("Competitor Price", ["Low", "Medium", "High"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
season = st.sidebar.selectbox("Season", ["Normal", "Festive"])

# Create input dataframe
input_data = pd.DataFrame({
    "Price_Change": [price_change],
    "Demand_Level": [demand],
    "Competitor_Price": [competitor],
    "Region": [region],
    "Season": [season]
})

# One-hot encode
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

if st.button("🔍 Analyze Decision"):
    prediction = model.predict(input_encoded)[0]

    st.subheader("📈 Decision Outcome Prediction")

    if prediction < 0:
        st.error(f"⚠️ Predicted Loss: {round(prediction,2)}")
    else:
        st.success(f"✅ Predicted Profit: {round(prediction,2)}")

    # SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)

    st.subheader("🔍 Feature Impact Explanation")

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        input_encoded,
        max_display=5,
        show=False
    )
    st.pyplot(fig)
# python -m venv venv
# venv\Scripts\activate
# python -m streamlit run app.py
