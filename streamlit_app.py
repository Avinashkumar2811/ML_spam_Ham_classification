import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"  # local fastapi

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

st.title("📩 SMS Spam Detection App")
st.write("Enter an SMS below to check whether it is **SPAM** or **HAM**.")

sms_input = st.text_area("✉️ Enter SMS message")

if st.button("Predict"):
    if sms_input.strip() == "":
        st.warning("Please enter an SMS")
    else:
        response = requests.post(
            API_URL,
            json={"sms": sms_input}
        )

        if response.status_code == 200:
            result = response.json()
            if result["prediction"] == "SPAM":
                st.error("🚨 SPAM MESSAGE")
            else:
                st.success("✅ HAM (Normal Message)")
        else:
            st.error("FastAPI server not running")
