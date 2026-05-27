import streamlit as st
import numpy as np
import joblib

model = joblib.load(
    "decision_tree_classifier.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

st.title(
    "Decision Tree Cancer Prediction"
)

mean_radius = st.number_input(
    "Mean Radius",
    10.0
)

mean_texture = st.number_input(
    "Mean Texture",
    10.0
)

mean_perimeter = st.number_input(
    "Mean Perimeter",
    100.0
)

mean_area = st.number_input(
    "Mean Area",
    500.0
)

mean_smoothness = st.number_input(
    "Mean Smoothness",
    0.1
)

if st.button(
    "Predict"
):

    features = np.zeros(
        (1,30)
    )

    features[0][0] = mean_radius
    features[0][1] = mean_texture
    features[0][2] = mean_perimeter
    features[0][3] = mean_area
    features[0][4] = mean_smoothness

    features = scaler.transform(
        features
    )

    prediction = model.predict(
        features
    )

    if prediction[0] == 1:

        st.success(
            "Benign"
        )

    else:

        st.error(
            "Malignant"
        )