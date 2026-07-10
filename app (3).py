import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Career Role Predictor", layout="wide")

st.title("🎯 Tech Career Role Predictor (Using Saved Model)")
st.write("Select your proficiency or interest level for each skill to see your predicted job role.")

# 2. Load the Pickled Model and Encoders
@st.cache_resource
def load_pickled_assets():
    # Make sure 'career_recommendation_model.pkl' is in the same directory
    with open("career_recommendation_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data['model'], data['feature_encoders'], data['target_encoder'], data['feature_names']

try:
    model, feature_encoders, target_encoder, feature_names = load_pickled_assets()
except FileNotFoundError:
    st.error("Error: 'career_recommendation_model.pkl' not found. Please train and save the model first.")
    st.stop()

# 3. Build the UI Form
with st.form("prediction_form"):
    st.subheader("Skill Profile")
    
    user_inputs = {}
    # Split the screen into 3 clean vertical columns
    cols = st.columns(3)
    
    # Iterate dynamically using the exact feature names extracted from the pickle file
    for i, col_name in enumerate(feature_names):
        with cols[i % 3]:
            # Pull the original categorical text choices ("Intermediate", "Not Interested", etc.)
            options = feature_encoders[col_name].classes_
            user_inputs[col_name] = st.selectbox(col_name, options)
            
    st.divider()
    submit = st.form_submit_button("Predict My Role", type="primary")

# 4. Handle Form Submission & Predictions
if submit:
    # Build the input DataFrame directly using the dictionary
    input_df = pd.DataFrame([user_inputs])
    
    # CRITICAL: Force the DataFrame to explicitly follow the training layout order
    input_df = input_df[feature_names]
    
    # Transform text selections to numerical values using the loaded encoders
    for col_name in input_df.columns:
        le = feature_encoders[col_name]
        input_df[col_name] = le.transform(input_df[col_name].astype(str))
        
    # Generate the class prediction mapping
    prediction_num = model.predict(input_df)
    
    # Decode back to the readable career role text
    predicted_role = target_encoder.inverse_transform(prediction_num)[0]
    
    # Display Output UI
    st.success(f"### Predicted Role: {predicted_role}")
    st.balloons()
