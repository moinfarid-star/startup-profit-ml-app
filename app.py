import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Startup Profit Prediction", layout="centered")

st.title("Startup Profit Prediction App")
st.write("Predict startup profit using Multiple Linear Regression")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("50_Startups.csv")

data = load_data()

# Features & target
X = data[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = data["Profit"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Startup Details")

rd = st.number_input("R&D Spend", min_value=0.0, value=150000.0)
admin = st.number_input("Administration Spend", min_value=0.0, value=120000.0)
marketing = st.number_input("Marketing Spend", min_value=0.0, value=300000.0)
state = st.selectbox("State", ["California", "Florida", "New York"])

# Prepare input
input_data = pd.DataFrame({
    "R&D Spend": [rd],
    "Administration": [admin],
    "Marketing Spend": [marketing],
    "State": [state]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

if st.button("Predict Profit"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Profit: ${prediction[0]:,.2f}")
