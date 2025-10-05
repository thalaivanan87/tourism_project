import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = os.getenv("HF_MODEL_REPO", "thalaivanan/tourism_model")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_tourism_model_v1.joblib")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "tourism_project/models/best_tourism_model_v1.joblib")

def load_model():
    # try to download from HF Hub first, fall back to local path
    try:
        model_file = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        model = joblib.load(model_file)
        return model
    except Exception:
        if os.path.exists(LOCAL_MODEL_PATH):
            return joblib.load(LOCAL_MODEL_PATH)
        raise RuntimeError("Model not found in HF Hub or local path.")

st.title("Tourism Booking Prediction")

st.markdown(
    """
Enter the guest and pitch details below and click **Predict** to get the booking prediction and probability.
"""
)

# Numeric inputs (defaults chosen as reasonable placeholders)
age = st.number_input("Age", min_value=0, max_value=120, value=35)
city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0.0, value=5.0, step=0.5)
num_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=1)
preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=1)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)

# Categorical inputs (free text to remain compatible with OneHotEncoder(handle_unknown='ignore'))
type_of_contact = st.text_input("Type of Contact", value="Self")
occupation = st.text_input("Occupation", value="Salaried")
gender = st.text_input("Gender", value="Male")
product_pitched = st.text_input("Product Pitched", value="A")
marital_status = st.text_input("Marital Status", value="Single")
designation = st.text_input("Designation", value="Manager")

# Build input DataFrame in the same column names used during training
input_df = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": preferred_property_star,
    "NumberOfTrips": number_of_trips,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "NumberOfChildrenVisiting": num_children_visiting,
    "TypeofContact": str(type_of_contact),
    "Occupation": str(occupation),
    "Gender": str(gender),
    "ProductPitched": str(product_pitched),
    "MaritalStatus": str(marital_status),
    "Designation": str(designation)
}])

st.write("Input preview:")
st.table(input_df)

if st.button("Predict"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
    else:
        # model is expected to be a pipeline that accepts raw DataFrame
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
            thresh = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.45))
            pred = int(prob >= thresh)
            label = "Booked" if pred == 1 else "Not Booked"
            st.subheader("Prediction")
            st.write(f"Prediction: **{label}**")
            st.write(f"Booking probability: **{prob:.3f}** ({prob*100:.1f}%)")
        else:
            # fallback to predict
            pred = model.predict(input_df)[0]
            label = "Booked" if int(pred) == 1 else "Not Booked"
            st.subheader("Prediction")
            st.write(f"Prediction: **{label}**")
