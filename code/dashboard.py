import streamlit as st
import pandas as pd
import pickle

# Load the model
@st.cache_resource  # Cache model for performance
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# UI Layout
st.title("PowerPredict: Energy Output Dashboard")
st.write("Enter power plant parameters to predict electrical output (PE) in MW.")

# Input fields
T = st.number_input("Temperature (T, Celsius)", min_value=0.0, max_value=50.0, value=25.0)
AP = st.number_input("Ambient Pressure (AP)", min_value=900.0, max_value=1100.0, value=1013.0)
RH = st.number_input("Relative Humidity (RH)", min_value=0.0, max_value=100.0, value=60.0)
V = st.number_input("Exhaust Vacuum (V)", min_value=20.0, max_value=80.0, value=40.0)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[T, AP, RH, V]], columns=['T', 'AP', 'RH', 'V'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Energy Output (PE): {prediction:.3f} MW")

# Optional: Add a note
st.write("*Model trained on CCPP data with RMSE ~3.67 MW. Temperature in Celsius.*")