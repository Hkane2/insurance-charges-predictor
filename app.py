import pandas as pd
import streamlit as st

from train_model import build_model, make_features

st.set_page_config(page_title="Insurance Charges Predictor", page_icon="💰", layout="centered")


@st.cache_resource
def load_model():
    # Trains once when the app starts, then is reused
    return build_model("insurance.csv")


model, test_r2, test_mae = load_model()

st.title("💰 Medical Insurance Charges Predictor")
st.caption(f"Linear Regression trained on 1,337 people · "
           f"test R² {test_r2:.2f} · average error ±${test_mae:,.0f}")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"], format_func=str.capitalize)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1,
                          help="Body Mass Index = weight (kg) / height (m)²")

with col2:
    children = st.number_input("Number of children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"], format_func=str.capitalize)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"],
                          format_func=str.capitalize)

if st.button("Predict charges", type="primary", use_container_width=True):
    person = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region,
    }])
    prediction = model.predict(make_features(person))[0]
    prediction = max(prediction, 0)  # charges can't be negative

    st.metric("Estimated yearly insurance charges", f"${prediction:,.0f}")

    if smoker == "yes" and bmi >= 30:
        st.warning("Smoking combined with a BMI of 30+ is the biggest cost driver in this data "
                   "(about +$19,500 compared with the same person without both).")
    elif smoker == "yes":
        st.info("Smoking adds about $13,000 per year in this data.")

st.divider()
st.caption("A machine-learning demo for learning purposes. "
           "Real insurance prices depend on many more factors.")
