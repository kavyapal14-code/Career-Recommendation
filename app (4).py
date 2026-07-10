import streamlit as st
import pickle
import numpy as np

# Load model
with open('career_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load role encoder
with open('le_role.pkl', 'rb') as f:
    le_role = pickle.load(f)

# Load skill scale + column order
with open('skill_map.pkl', 'rb') as f:
    skill_data = pickle.load(f)

skill_order = skill_data["skill_order"]
skill_map = skill_data["skill_map"]
columns = skill_data["columns"]

st.title("IT Career Role Recommender")

st.write("Rate your skill / interest level in each area below:")

col1, col2 = st.columns(2)
responses = {}

for i, skill in enumerate(columns):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        responses[skill] = st.selectbox(skill, skill_order, index=0)

# Encode responses in the same order the model was trained on
features = np.array([[skill_map[responses[col]] for col in columns]])

if st.button("Predict My Career Role"):
    prediction = model.predict(features)
    role = le_role.inverse_transform(prediction)[0]
    st.success(f"Recommended career role: {role}")
