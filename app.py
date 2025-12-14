import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------- Page Config ----------
st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="📈",
    layout="wide"
)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("50_Startups.csv")

data = load_data()

# ---------- Prepare Data ----------
X_raw = data[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = data["Profit"]

X = pd.get_dummies(X_raw, drop_first=True)

# Train/Test (optional, for score display)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
test_r2 = model.score(X_test, y_test)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
show_data = st.sidebar.checkbox("Show dataset preview", value=False)
show_about = st.sidebar.checkbox("Show model details", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use realistic spends for better results.")

# ---------- Header ----------
st.title("📈 Startup Profit Prediction App")
st.write("Predict startup profit using **Multiple Linear Regression** (with one-hot encoding for State).")

# ---------- Layout ----------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🧾 Enter Startup Details")

    # Quick-fill button
    if st.button("✨ Use Example Values"):
        st.session_state["rd"] = 150000.0
        st.session_state["admin"] = 120000.0
        st.session_state["marketing"] = 300000.0
        st.session_state["state"] = "California"

    rd = st.number_input("R&D Spend", min_value=0.0, value=st.session_state.get("rd", 150000.0), step=1000.0, format="%.2f")
    admin = st.number_input("Administration Spend", min_value=0.0, value=st.session_state.get("admin", 120000.0), step=1000.0, format="%.2f")
    marketing = st.number_input("Marketing Spend", min_value=0.0, value=st.session_state.get("marketing", 300000.0), step=1000.0, format="%.2f")

    states = sorted(data["State"].unique().tolist())
    state = st.selectbox("State", states, index=states.index(st.session_state.get("state", states[0])) if st.session_state.get("state", states[0]) in states else 0)

    predict = st.button("🚀 Predict Profit", type="primary")

with right:
    st.subheader("📊 Output")

    # Build input row
    input_df = pd.DataFrame({
        "R&D Spend": [rd],
        "Administration": [admin],
        "Marketing Spend": [marketing],
        "State": [state]
    })

    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    if predict:
        pred = model.predict(input_encoded)[0]

        st.success("Prediction completed ✅")

        # Pretty metrics
        col1, col2 = st.columns(2)
        col1.metric("Predicted Profit", f"${pred:,.2f}")
        col2.metric("Test R² Score", f"{test_r2:.3f}")

        st.markdown("#### 🧠 Notes")
        st.caption("Prediction is based on historical data and linear assumptions. Use it for learning/demo purposes.")

    else:
        st.info("Enter values and click **Predict Profit** to see the result.")

# ---------- Optional sections ----------
st.markdown("---")

if show_about:
    st.subheader("ℹ️ Model Details")
    st.write(
        """
- Algorithm: **Multiple Linear Regression**
- Categorical handling: **One-Hot Encoding (State)**
- Train/Test split: **80/20**
        """
    )

if show_data:
    st.subheader("🗂️ Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)
