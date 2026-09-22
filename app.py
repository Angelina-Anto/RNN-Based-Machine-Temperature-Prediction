import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_excel("RNN_Machine_Temperature.xlsx")

features = df[["Temperature", "Vibration"]].values


# -----------------------------
# Create the same scalers
# used during model training
# -----------------------------
scaler_X = MinMaxScaler()
scaler_X.fit(features)

scaler_y = MinMaxScaler()
scaler_y.fit(df[["Temperature"]])


# -----------------------------
# Load saved RNN model
# -----------------------------
model = tf.keras.models.load_model("machine_temperature_rnn.keras")


# -----------------------------
# Streamlit title
# -----------------------------
st.title("Machine Temperature Predictor")


# -----------------------------
# User inputs
# -----------------------------
st.subheader("Previous Timestamp 1")

temperature_1 = st.number_input(
    "Temperature",
    value=80.0
)

vibration_1 = st.number_input(
    "Vibration",
    value=3.5
)


st.subheader("Previous Timestamp 2")

temperature_2 = st.number_input(
    "Temperature ",
    value=82.0
)

vibration_2 = st.number_input(
    "Vibration ",
    value=3.6
)


# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Next Temperature"):

    # Create input data
    new_readings = np.array([
        [temperature_1, vibration_1],
        [temperature_2, vibration_2]
    ])

    # Scale input using the same scaler
    new_readings_scaled = scaler_X.transform(new_readings)

    # Add batch dimension
    new_input = np.array([
        new_readings_scaled
    ])

    # Predict
    predicted_scaled = model.predict(new_input, verbose=0)

    # Convert prediction back to actual temperature
    predicted_temperature = scaler_y.inverse_transform(
        predicted_scaled
    )

    # Display result
    st.success(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature[0][0]:.2f} °C"
    )
