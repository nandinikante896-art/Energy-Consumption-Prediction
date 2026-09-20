import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)


# =====================================================
# PAGE TITLE
# =====================================================

st.title("⚡ Energy Consumption Predictor")

st.caption(
    "AI-Powered Building Energy Consumption Prediction System"
)

st.divider()


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = Path("models/energy_consumption_model.pkl")

if not MODEL_PATH.exists():

    st.error("❌ Model file not found!")

    st.stop()


model = joblib.load(MODEL_PATH)


# =====================================================
# INPUT SECTION
# =====================================================

st.header("🏢 Building Information")


col1, col2 = st.columns(2)


# =====================================================
# LEFT COLUMN
# =====================================================

with col1:

    temperature = st.number_input(
        "🌡️ Temperature",
        min_value=-20.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

    square_footage = st.number_input(
        "📐 Square Footage",
        min_value=100.0,
        max_value=100000.0,
        value=2000.0,
        step=100.0
    )

    occupancy = st.number_input(
        "👥 Occupancy",
        min_value=0,
        max_value=5000,
        value=50,
        step=1
    )

    renewable_energy = st.number_input(
        "☀️ Renewable Energy",
        min_value=0.0,
        max_value=100000.0,
        value=500.0,
        step=50.0
    )


# =====================================================
# RIGHT COLUMN
# =====================================================

with col2:

    hvac_usage = st.selectbox(
        "❄️ HVAC Usage",
        ["Low", "Medium", "High"]
    )

    lighting_usage = st.selectbox(
        "💡 Lighting Usage",
        ["Low", "Medium", "High"]
    )

    day_of_week = st.selectbox(
        "📅 Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    holiday = st.selectbox(
        "🎉 Holiday",
        ["No", "Yes"]
    )

    selected_date = st.date_input(
        "📅 Date",
        value=datetime.today()
    )

    hour = st.slider(
        "🕐 Hour of Day",
        0,
        23,
        12
    )


st.divider()


# =====================================================
# PREDICTION BUTTON
# =====================================================

predict_button = st.button(
    "⚡ Predict Energy Consumption",
    type="primary",
    use_container_width=True
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    # Date features

    year = selected_date.year
    month = selected_date.month
    day = selected_date.day


    # -------------------------------------------------
    # CREATE INPUT DATA
    # -------------------------------------------------

    input_data = pd.DataFrame({

        "Temperature": [float(temperature)],

        "Humidity": [float(humidity)],

        "SquareFootage": [float(square_footage)],

        "Occupancy": [float(occupancy)],

        "HVACUsage": [hvac_usage],

        "LightingUsage": [lighting_usage],

        "RenewableEnergy": [float(renewable_energy)],

        "DayOfWeek": [day_of_week],

        "Year": [int(year)],

        "Month": [int(month)],

        "Day": [int(day)],

        "Hour": [int(hour)],

        "Holiday": [holiday]
    })


    # -------------------------------------------------
    # PREDICT
    # -------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]


        # =============================================
        # RESULT
        # =============================================

        st.divider()

        st.header("🎯 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                label="⚡ Energy Consumption",
                value=f"{prediction:.2f}"
            )

        with col2:

            st.metric(
                label="🌡️ Temperature",
                value=f"{temperature:.1f}"
            )

        with col3:

            st.metric(
                label="👥 Occupancy",
                value=str(occupancy)
            )


        st.success(
            f"✅ Predicted Energy Consumption: {prediction:.2f}"
        )


        # =============================================
        # INPUT DETAILS
        # =============================================

        with st.expander("🔍 View Input Details"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🤖 Machine Learning • Energy Efficiency Prediction System • 2026"
)