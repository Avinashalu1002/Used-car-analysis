import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ---------------- LOAD MODEL ----------------

model = joblib.load("xgb_model.pkl")

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Model Info")
st.sidebar.write("**Algorithm:** XGBoost")
st.sidebar.write("**Features Used:**")
st.sidebar.write("- Brand")
st.sidebar.write("- Model")
st.sidebar.write("- City")
st.sidebar.write("- Fuel")
st.sidebar.write("- Transmission")
st.sidebar.write("- Car Age")
st.sidebar.write("- KMS Driven")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("car_data.csv")
df.columns = df.columns.str.replace('"', '').str.strip()

# Fix broken dataset (if needed)

if len(df.columns) == 1:
df = df[df.columns[0]].str.split(",", expand=True)
df.columns = ["index","Brand","Car","Year","City","KMS","Fuel","Transmission","Price","Emi"]

# Drop unnecessary columns

if "index" in df.columns:
df.drop("index", axis=1, inplace=True)
if "Emi" in df.columns:
df.drop("Emi", axis=1, inplace=True)

# Convert numeric

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["KMS"] = pd.to_numeric(df["KMS"], errors="coerce")

# ---------------- FEATURE ENGINEERING ----------------

df["Model"] = df["Car"].apply(lambda x: x.split()[0])

# ---------------- UI ----------------

col1, col2 = st.columns(2)

with col1:
brand = st.selectbox("Select Brand", sorted(df["Brand"].dropna().unique()))

```
filtered_models = df[df["Brand"] == brand]["Model"].dropna().unique()
model_name = st.selectbox("Select Model", sorted(filtered_models))

city = st.selectbox("Select City", sorted(df["City"].dropna().unique()))
```

with col2:
fuel = st.selectbox("Fuel Type", sorted(df["Fuel"].dropna().unique()))
transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
year = st.number_input("Year", 2000, datetime.now().year, 2018)
kms = st.number_input("KMS Driven", 0, 200000, 30000)

# ---------------- CREATE FEATURES ----------------

car_age = datetime.now().year - year

# ---------------- PREDICTION ----------------

st.write("")

if st.button("🔮 Predict Price", use_container_width=True):

```
input_data = pd.DataFrame([{
    "Brand": brand,
    "City": city,
    "Fuel": fuel,
    "Transmission": transmission,
    "Model": model_name,
    "KMS": kms,
    "Car_Age": car_age
}])

# 🔥 Correct prediction (log → real price)
prediction = model.predict(input_data)[0]
prediction = np.expm1(prediction)

# Optional range
low = int(prediction * 0.9)
high = int(prediction * 1.1)

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
    <h2>💰 Estimated Price</h2>
    <h1 style='color: green;'>₹ {int(prediction):,}</h1>
    <p>Expected range: ₹ {low:,} - ₹ {high:,}</p>
</div>
""", unsafe_allow_html=True)
```

# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown(
"<p style='text-align: center;'>Provide the car specifications to predict its expected market price.</p>",
unsafe_allow_html=True
)


