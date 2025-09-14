import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page settings
st.set_page_config(page_title="Smart Irrigation Prediction", layout="centered")

# Title
st.title("🌱 Smart Irrigation Prediction using AI/ML")

st.write("This app predicts the *water requirement* for different crops and gives easy-to-understand advice for farmers.")

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("📥 Enter Crop & Environmental Parameters")

# Crop selection
crop = st.selectbox("Select Crop", ["Wheat", "Rice", "Maize"])

soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=30)
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=60, value=25)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)

# -------------------------------
# Dummy Training Data
# -------------------------------
if crop == "Wheat":
    X = np.array([[20, 20, 40], [30, 22, 50], [40, 25, 55], [50, 28, 60], [60, 30, 65]])
    y = np.array([30, 50, 70, 90, 110])
elif crop == "Rice":
    X = np.array([[20, 25, 60], [30, 28, 65], [40, 30, 70], [50, 32, 75], [60, 35, 80]])
    y = np.array([50, 70, 90, 110, 130])
else:  # Maize
    X = np.array([[20, 23, 45], [30, 25, 55], [40, 28, 60], [50, 30, 65], [60, 33, 70]])
    y = np.array([40, 60, 85, 105, 125])

# Train model
model = LinearRegression().fit(X, y)

# Model accuracy
y_pred = model.predict(X)
accuracy = r2_score(y, y_pred) * 100

# -------------------------------
# Prediction
# -------------------------------
if st.button("💧 Predict Water Requirement"):
    pred = model.predict([[soil_moisture, temperature, humidity]])[0]
    
    # Show farmer-friendly result
    st.success(f"### ✅ For {crop}, you should give around *{pred:.0f} liters of water* per acre.")
    
    # Farmer-friendly advice
    if pred < 50:
        advice = "💡 Soil has enough moisture. Give *less water* to avoid wastage."
    elif 50 <= pred <= 100:
        advice = "💡 Give *moderate water*. This is the optimal range for crop growth."
    else:
        advice = "💡 Crop needs *more water* due to high temperature and low soil moisture."
    
    st.warning(advice)
    
    # Show accuracy
    st.info(f"📊 Model Accuracy: {accuracy:.2f}% (based on training data)")
    
    # -------------------------------
    # Graph: Show only Prediction Trend (Farmer-friendly)
    # -------------------------------
    st.subheader("📈 Water Requirement Trend for Training Samples")

    fig, ax = plt.subplots()
    ax.plot(y, label="Ideal Water Requirement", marker="o")
    ax.plot(y_pred, label="Model Prediction", marker="x", linestyle="--")
    ax.axhline(pred, color="red", linestyle=":", label=f"Your Crop Prediction ({pred:.0f})")
    
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Water Requirement (liters/acre)")
    ax.set_ylim(0, max(max(y), max(y_pred), pred) + 20)  # consistent scale
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# -------------------------------
# Extra Information
# -------------------------------
st.caption("save Water, save Future! 🌍💧")
