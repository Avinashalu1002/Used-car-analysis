import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model
model = joblib.load("xgbs_model.pkl")

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Model Info")
st.sidebar.write("Algorithm: XGBoost")
st.sidebar.write("Features: Brand, Model, City, Fuel, Transmission, Car Age, KMS")

# Load data
df = pd.read_csv("car_data.csv")
df.columns = df.columns.str.replace('"', '').str.strip()

# Fix dataset if needed
if len(df.columns) == 1:
    df = df[df.columns[0]].str.split(",", expand=True)
    df.columns = ["index","Brand","Car","Year","City","KMS","Fuel","Transmission","Price","Emi"]

# Drop unwanted columns
if "index" in df.columns:
    df.drop("index", axis=1, inplace=True)

if "Emi" in df.columns:
    df.drop("Emi", axis=1, inplace=True)

# Convert types
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["KMS"] = pd.to_numeric(df["KMS"], errors="coerce")

# Feature engineering
df["Model"] = df["Car"].apply(lambda x: x.split()[0])

# UI layout
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Brand", sorted(df["Brand"].dropna().unique()))
    filtered_models = df[df["Brand"] == brand]["Model"].dropna().unique()
    model_name = st.selectbox("Select Model", sorted(filtered_models))
    city = st.selectbox("Select City", sorted(df["City"].dropna().unique()))

with col2:
    fuel = st.selectbox("Fuel Type", sorted(df["Fuel"].dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
    year = st.number_input("Year", 2000, datetime.now().year, 2018)
    kms = st.number_input("KMS Driven", 0, 200000, 30000)

# Create feature
car_age = datetime.now().year - year

# Prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "Brand": brand,
        "City": city,
        "Fuel": fuel,
        "Transmission": transmission,
        "Model": model_name,
        "KMS": kms,
        "Car_Age": car_age
    }])

    prediction = model.predict(input_data)[0]
    prediction = np.expm1(prediction)

    st.markdown("---")
    st.write("### Estimated Price")
    st.success(f"₹ {int(prediction):,}")

    low = int(prediction * 0.9)
    high = int(prediction * 1.1)
    st.write(f"Expected range: ₹ {low:,} - ₹ {high:,}")

# Footer
st.markdown("---")
st.write("Provide car details to predict price.")


