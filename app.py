import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

st.title("🌾 Machine Learning Based Crop Yield Prediction System")
st.markdown("Predict crop yield using Random Forest model")

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("crop_yield_rf_model.pkl")

model = load_model()

# ---------------------------------------------------
# Dropdown Data
# ---------------------------------------------------

states = [
'Assam','Karnataka','Kerala','Meghalaya','West Bengal','Puducherry','Goa',
'Andhra Pradesh','Tamil Nadu','Odisha','Bihar','Gujarat','Madhya Pradesh',
'Maharashtra','Mizoram','Punjab','Uttar Pradesh','Haryana',
'Himachal Pradesh','Tripura','Nagaland','Chhattisgarh','Uttarakhand',
'Jharkhand','Delhi','Manipur','Jammu and Kashmir','Telangana',
'Arunachal Pradesh','Sikkim'
]

crops = [
'Arecanut','Arhar/Tur','Castor seed','Coconut','Cotton(lint)',
'Dry chillies','Gram','Jute','Linseed','Maize','Mesta','Niger seed',
'Onion','Other  Rabi pulses','Potato','Rapeseed &Mustard','Rice',
'Sesamum','Small millets','Sugarcane','Sweet potato','Tapioca','Tobacco',
'Turmeric','Wheat','Bajra','Black pepper','Cardamom','Coriander','Garlic',
'Ginger','Groundnut','Horse-gram','Jowar','Ragi','Cashewnut','Banana',
'Soyabean','Barley','Khesari','Masoor','Moong(Green Gram)',
'Other Kharif pulses','Safflower','Sannhamp','Sunflower','Urad',
'Peas & beans (Pulses)','other oilseeds','Other Cereals','Cowpea(Lobia)',
'Oilseeds total','Guar seed','Other Summer Pulses','Moth'
]

seasons = ['Kharif','Rabi','Whole Year']

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

st.header("Enter Crop Details")

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Select Crop", sorted(crops))
    state = st.selectbox("Select State", sorted(states))
    season = st.selectbox("Select Season", seasons)

with col2:
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)
    fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, value=50000.0)
    pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=2000.0)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, value=25.0)

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------

if st.button("Predict Yield"):

    try:
        input_df = pd.DataFrame({
            "Crop": [crop],
            "Season": [season],
            "State": [state],
            "Annual_Rainfall": [annual_rainfall],
            "Fertilizer": [fertilizer],
            "Pesticide": [pesticide],
            "Avg_Temperature": [avg_temp]
        })

        # One Hot Encoding
        input_df = pd.get_dummies(input_df)

        # Align columns with model training features
        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Predict log value
        prediction_log = model.predict(input_df)

        # Convert back from log
        tons_per_hectare = np.expm1(prediction_log)[0]

        # Convert to quintal per acre
        quintal_per_acre = tons_per_hectare * 4.04686

        # ---------------------------------------------------
        # Display Results
        # ---------------------------------------------------

        st.success("🌱 Yield Prediction Results")

        st.write(f"### 📌 Tons per Hectare (t/ha): {tons_per_hectare:.3f}")
        st.write(f"### 📌 Quintal per Acre (q/acre): {quintal_per_acre:.3f}")

        # Yield Interpretation
        if tons_per_hectare < 1:
            st.warning("⚠️ Low yield expected. Consider improving irrigation, fertilizer usage, or crop conditions.")
        elif tons_per_hectare < 3:
            st.info("✅ Moderate yield expected under current conditions.")
        else:
            st.success("🌾 High yield expected. Conditions seem very favorable!")

    except Exception as e:
        st.error(f"Error occurred: {e}")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")
st.markdown("Built by Shyam 🚀 | Random Forest Model | Test R² ≈ 0.96")