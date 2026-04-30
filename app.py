import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("best_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Model Info")
st.sidebar.write("**Algorithm:** XGBoost")
st.sidebar.write("**R² Score:** 0.85")
st.sidebar.write("**MAE:** ₹1,06,880")
st.sidebar.write("---")
st.sidebar.write("**Features Used:**")
st.sidebar.write("- Brand")
st.sidebar.write("- City")
st.sidebar.write("- Fuel")
st.sidebar.write("- Transmission")
st.sidebar.write("- Car Age")
st.sidebar.write("- KMS Driven")

# ---------------- LOAD & FIX DATASET ----------------
df = pd.read_csv("car_data.csv", header=0)

# Clean column names
df.columns = df.columns.str.replace('"', '').str.strip()

# If dataset is broken into one column → fix it
if len(df.columns) == 1:
    df = df[df.columns[0]].str.split(",", expand=True)
    df.columns = ["index","Brand","Car","Year","City","KMS","Fuel","Transmission","Price","Emi"]

# Drop unwanted index column
if "index" in df.columns:
    df.drop("index", axis=1, inplace=True)

# Convert numeric columns
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["KMS"] = pd.to_numeric(df["KMS"], errors="coerce")

# ---------------- DROPDOWNS ----------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Brand", sorted(df["Brand"].dropna().unique()))
    city = st.selectbox("Select City", sorted(df["City"].dropna().unique()))
    fuel = st.selectbox("Fuel Type", sorted(df["Fuel"].dropna().unique()))

with col2:
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
    year = st.number_input("Year", 2000, 2025, 2018)
    kms = st.number_input("KMS Driven", 0, 200000, 30000)

# ---------------- FEATURE ENGINEERING ----------------
car_age = 2025 - year

# ---------------- PREDICTION ----------------
st.write("")

if st.button("🔮 Predict Price", use_container_width=True):

    input_data = pd.DataFrame([{
        "Brand": brand,
        "City": city,
        "Fuel": fuel,
        "Transmission": transmission,
        "Car_Age": car_age,
        "KMS": kms
    }])

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
        <h2>💰 Estimated Price</h2>
        <h1 style='color: green;'>₹ {int(prediction):,}</h1>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ using Streamlit | ML Model: XGBoost</p>",
    unsafe_allow_html=True
)
