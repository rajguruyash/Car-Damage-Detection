import streamlit as st
from model_helper import predict

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload a file", type = ['jpg','png'])
if uploaded_file:
    imagepath = "temp_file.jpg"
    with open(imagepath, "wb") as f:
        f.write(uploaded_file.read())
        prediction = predict(imagepath)
        st.image(uploaded_file, caption="Uploaded File", use_column_width=True)
        st.info(f"Prediction: {prediction}")