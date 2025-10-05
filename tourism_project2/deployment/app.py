# tourism_project/deployment/app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Config (can override via env vars if you prefer)
MODEL_REPO = os.getenv("HF_MODEL_REPO", "thalaivanan/tourism_model")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_tourism_model_v1.joblib")
# default relative local path (we will compute absolute path at runtime)
DEFAULT_LOCAL_REL = f"tourism_project/models/{MODEL_FILENAME}"

def load_model():
    """
    Try local model first (LOCAL_MODEL_PATH env or DEFAULT_LOCAL_REL),
    then try HF Hub (use HUGGINGFACE_HUB_TOKEN env if set).
    Raises RuntimeError with clear message on failure.
    """
    # determine local path at call time (this allows setting env var from UI before Predict)
    local_path_str = os.getenv("LOCAL_MODEL_PATH", DEFAULT_LOCAL_REL)
    local_path = Path(local_path_str)
    # prefer absolute path relative to working dir if not absolute
    if not local_path.is_absolute():
        # try resolve relative to common Colab path /content first
        candidate = Path("/content") / local_path
        if candidate.exists():
            local_path = candidate
        else:
            # fallback to the given relative path (so it still works locally)
            local_path = Path(local_path_str)

    # 1) try local
    if local_path.exists():
        try:
            return joblib.load(local_path)
        except Exception as e:
            raise RuntimeError(f"Found local model at {local_path} but failed to load: {e}")

    # 2) try HF Hub (use token if provided in env)
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", os.getenv("HF_TOKEN", None))
    try:
        model_file = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=hf_token)
        return joblib.load(model_file)
    except Exception as e:
        # surface a clear error that includes the cause (but not tokens)
        raise RuntimeError(
            f"Could not load model from local ({local_path}) or HF repo ({MODEL_REPO}). "
            f"Last error: {e}"
        )

st.title("Tourism Booking Prediction")
st.markdown("Enter guest and pitch details and click **Predict** to get a booking prediction and probability.")

# --- Inputs (unchanged)
age = st.number_input("Age", min_value=0, max_value=120, value=35)
city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0.0, value=5.0, step=0.5)
num_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=1)
preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=1)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)

type_of_contact = st.text_input("Type of Contact", value="Self")
occupation = st.text_input("Occupation", value="Salaried")
gender = st.text_input("Gender", value="Male")
product_pitched = st.text_input("Product Pitched", value="A")
marital_status = st.text_input("Marital Status", value="Single")
designation = st.text_input("Designation", value="Manager")

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

st.markdown("---")
st.markdown("### Model options (use one of these if loading fails)")
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload local model (.joblib or .pkl)", type=["joblib", "pkl"], key="model_upload")
    if uploaded is not None:
        # save uploaded file into models folder and set LOCAL_MODEL_PATH for this session
        models_dir = Path("/content/tourism_project/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        dst = models_dir / MODEL_FILENAME
        with open(dst, "wb") as f:
            f.write(uploaded.getbuffer())
        # set env var for this process so load_model() will pick it up
        os.environ["LOCAL_MODEL_PATH"] = str(dst)
        st.success(f"Uploaded model and set LOCAL_MODEL_PATH -> {dst}")

with col2:
    hf_token_input = st.text_input("Hugging Face token (if repo private)", type="password", help="Paste a HF token only for this session; don't hard-code it.")
    if hf_token_input:
        # store in env for load_model to use during this session (not persisted)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token_input
        st.success("HF token set for this session (will be used for HF downloads).")

# Prediction
if st.button("Predict"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Fixes: (1) Upload the .joblib model above, or (2) paste a Hugging Face token (if the HF repo is private), "
                "or set LOCAL_MODEL_PATH env var to the absolute path of your model before starting Streamlit.")
    else:
        # model is expected to be a pipeline that accepts DataFrame
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
            thresh = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.45))
            pred = int(prob >= thresh)
            label = "Booked" if pred == 1 else "Not Booked"
            st.subheader("Prediction")
            st.write(f"Prediction: **{label}**")
            st.write(f"Booking probability: **{prob:.3f}** ({prob*100:.1f}%)")
        else:
            pred = model.predict(input_df)[0]
            label = "Booked" if int(pred) == 1 else "Not Booked"
            st.subheader("Prediction")
            st.write(f"Prediction: **{label}**")
