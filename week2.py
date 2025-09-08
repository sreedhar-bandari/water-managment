import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Irrigation Prediction", layout="centered")
st.title("🌱 Smart Irrigation Prediction (Text Output)")

# --- Generate dataset ---
@st.cache_data
def generate_dataset(seed: int = 42, n: int = 200):
    np.random.seed(seed)
    data = {
        "Soil_Moisture": np.random.randint(10, 90, n),
        "Temperature": np.random.randint(15, 40, n),
        "Rainfall": np.random.randint(0, 60, n),
        "Crop": np.random.choice(["Wheat", "Rice", "Maize"], n)
    }
    df = pd.DataFrame(data)
    df["Water_Needed"] = (
        (100 - df["Soil_Moisture"]) * 0.45 +
        (df["Temperature"] - 18) * 1.8 -
        df["Rainfall"] * 0.35 +
        np.where(df["Crop"] == "Rice", 22,
                 np.where(df["Crop"] == "Wheat", 12, 6))
    )
    df["Water_Needed"] = df["Water_Needed"] + np.random.normal(0, 2, size=n)
    return df.round(2)

# --- Prepare model ---
@st.cache_data
def train_model(df):
    le = LabelEncoder()
    df["Crop_Code"] = le.fit_transform(df["Crop"])
    X = df[["Soil_Moisture", "Temperature", "Rainfall", "Crop_Code"]]
    y = df["Water_Needed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, le

# --- Generate dataset and model ---
df = generate_dataset()
model, le = train_model(df)

# --- User input ---
st.subheader("Enter Parameters")
soil = st.slider("Soil Moisture (%)", 0, 100, 40)
temp = st.slider("Temperature (°C)", 0, 50, 30)
rain = st.slider("Rainfall (mm)", 0, 200, 5)
crop = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize"])

# --- Prediction ---
if st.button("Predict Water Requirement"):
    crop_code = le.transform([crop])[0]
    sample = np.array([[soil, temp, rain, crop_code]])
    prediction = model.predict(sample)[0]

    # --- Display output in text form ---
    st.write("### ✅ Prediction (Text Form)")
    st.write(f"Soil Moisture = {soil} %")
    st.write(f"Temperature = {temp} °C")
    st.write(f"Rainfall = {rain} mm")
    st.write(f"Crop = {crop}")
    st.write(f"➡️ Estimated Water Requirement = {prediction:.2f} units")
