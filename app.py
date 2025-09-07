import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained and optimized model
try:
    model = joblib.load('optimized_xgb_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'optimized_xgb_model.pkl' is in the same folder.")

# List all the features your model was trained on, in the exact same order
# This list was generated from your X_train DataFrame
training_features = ['Temperature', 'Humidity', 'hour', 'day_of_week', 'day_of_year',
                     'Shipment_Status_Delivered', 'Shipment_Status_In Transit',
                     'Traffic_Status_Detour', 'Traffic_Status_Heavy',
                     'Logistics_Delay_Reason_No Delay Reason', 'Logistics_Delay_Reason_Traffic',
                     'Logistics_Delay_Reason_Weather']


# This is a helper function to create the input DataFrame from user selections
def create_input_df(temp, humidity, hour, day_of_week, day_of_year,
                    shipment_status, traffic_status, logistics_delay_reason):
    
    # Create an empty DataFrame with all expected columns initialized to 0
    input_dict = {feature: [0] for feature in training_features}
    
    # Fill in the numerical values based on user input
    input_dict['Temperature'] = [temp]
    input_dict['Humidity'] = [humidity]
    input_dict['hour'] = [hour]
    input_dict['day_of_week'] = [day_of_week]
    input_dict['day_of_year'] = [day_of_year]

    # Fill in the one-hot encoded values based on user selections
    if shipment_status == 'Delivered': input_dict['Shipment_Status_Delivered'] = 1
    if shipment_status == 'In Transit': input_dict['Shipment_Status_In Transit'] = 1
    
    if traffic_status == 'Detour': input_dict['Traffic_Status_Detour'] = 1
    if traffic_status == 'Heavy': input_dict['Traffic_Status_Heavy'] = 1
    
    if logistics_delay_reason == 'No Delay Reason': input_dict['Logistics_Delay_Reason_No Delay Reason'] = 1
    if logistics_delay_reason == 'Traffic': input_dict['Logistics_Delay_Reason_Traffic'] = 1
    if logistics_delay_reason == 'Weather': input_dict['Logistics_Delay_Reason_Weather'] = 1

    return pd.DataFrame(input_dict)


# --- Streamlit Web App Layout ---
st.set_page_config(page_title="Cold Chain Spoilage Predictor")
st.title("🧊 AI-Powered Cold Chain Spoilage Predictor")
st.write("Enter the shipment details below to predict the likelihood of spoilage.")

# Create input fields for the user
st.header("Shipment Conditions")

col1, col2 = st.columns(2)
with col1:
    temp = st.slider("Temperature (°C)", min_value=-5.0, max_value=40.0, value=15.0, step=0.1, help="Temperature is the most important factor for spoilage.")
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    
with col2:
    shipment_status = st.selectbox("Shipment Status", ['Delivered', 'In Transit', 'On Hold', 'Delayed'])
    traffic_status = st.selectbox("Traffic Status", ['Clear', 'Moderate', 'Heavy', 'Detour'])
    logistics_delay_reason = st.selectbox("Delay Reason", ['No Delay Reason', 'Traffic', 'Weather', 'Mechanical Failure'])

# Logic for making the prediction when the button is clicked
if st.button("Predict Spoilage"):
    # Note: For a real app, you would get the current time to fill these features
    current_hour = pd.Timestamp.now().hour
    current_day_of_week = pd.Timestamp.now().dayofweek
    current_day_of_year = pd.Timestamp.now().dayofyear
    
    # Format the user's input into a DataFrame using the helper function
    input_df = create_input_df(temp, humidity, current_hour, current_day_of_week, current_day_of_year, 
                                shipment_status, traffic_status, logistics_delay_reason)
    
    # Get the prediction from the model
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("🚨 Prediction: The product is at high risk of SPOILAGE!")
    else:
        st.success("✅ Prediction: The product is likely NOT at risk of spoilage.")