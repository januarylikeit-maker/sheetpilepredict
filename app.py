import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained Random Forest model dictionary
@st.cache_resource
def load_model_data():
    return joblib.load('sheet_pile_rfr_model.pkl')

model_data = load_model_data()
model = model_data['model']
use_log = model_data['use_log']
feature_names = model_data['feature_names']

st.title("Sheet Pile Service Life Predictor")
st.write("Enter the primary site conditions below to predict the service life of the sheet piles.")

# User Interface Inputs
st.header("Site Conditions")

# Dropdown updated to only include the exact soils your model recognizes
uscs_soil = st.selectbox(
    "USCS Soil Classification", 
    ['CL', 'GC', 'GM', 'GP', 'GW', 'MH', 'ML', 'SC', 'SM', 'SP', 'SW']
)

flange_thick = st.slider("Flange Initial Thick. (mm)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)
corrosion_rate = st.slider("Factored Corrosion Rate (mm/yr)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
surcharge = st.slider("Surcharge (kPa)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
embedment = st.slider("Embedment Depth (m)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
gw_table = st.slider("Groundwater Table Elev. from Surface (m)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

# Backend Data Processing
# 1. Encode the USCS category using your exact console output
uscs_encoding_map = {
    'CL': 0, 'GC': 1, 'GM': 2, 'GP': 3, 'GW': 4, 
    'MH': 5, 'ML': 6, 'SC': 7, 'SM': 8, 'SP': 9, 'SW': 10
}

# The get function defaults to 9 (SP) if something unexpected occurs
encoded_uscs = uscs_encoding_map.get(uscs_soil, 9) 

# 2. Hardcode the baseline values for the hidden features
baseline_values = {
    'Lateral Effective Stress (kPa)': 45.0,
    'Vertical Effective Stress (kPa)': 120.0,
    'SPT Corrected N-Values': 15.0,
    'Internal Friction Angle (deg)': 30.0,
    'Effective Cohesion (kPa)': 10.0,
    'Saturated Unit Weight (kN/m³)': 18.5,
    'Mean Chloride Deposition': 0.5,
    'Mean Sulfate Deposition': 0.2,
    'Mean Humidity (%)': 80.0,
    'Mean Annual Temp (°C)': 28.0,
    'Void Ratio': 0.6,
    'Porosity': 0.4
}

if st.button("Calculate Predicted Service Life"):
    
    # 3. Assemble all user inputs
    input_dict = {
        'Soil_Type': encoded_uscs,
        'Flange Initial Thick. (mm)': flange_thick,
        'Factored Corrosion Rate (mm/yr)': corrosion_rate,
        'Surcharge (kPa)': surcharge,
        'Embedment Depth (m)': embedment,
        'Groundwater Table Elev. from Surface (m)': gw_table
    }
    
    # 4. Merge user inputs with the hardcoded baselines
    input_dict.update(baseline_values)
    
    # 5. Create DataFrame and enforce the exact column order from the trained model
    input_data = pd.DataFrame([input_dict])
    
    # This crucial line ensures the columns match the exact order of the training data
    input_data = input_data[feature_names] 
    
    # Generate prediction
    prediction_raw = model.predict(input_data)[0]
    
    # Back transform the prediction if the original model used log1p
    if use_log:
        predicted_years = np.expm1(prediction_raw)
    else:
        predicted_years = prediction_raw
        
    st.success(f"### Predicted Service Life: {round(predicted_years, 2)} years")