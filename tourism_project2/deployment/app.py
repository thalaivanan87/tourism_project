# tourism_project/deployment/app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration (override via env vars if needed)
MODEL_REPO = os.getenv("HF_MODEL_REPO", "thalaivanan/tourism_model")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "best_tourism_model_v1.joblib")

# Default absolute path for Colab / typical notebook runs.
# Override by setting LOCAL_MODEL_PATH env var (recommended in Colab before starting Streamlit).
DEFAULT_PROJECT_ROOT = Path("/content/tourism_project")  # change if your project root differs
DEFAULT_LOCAL_MODEL = DEFAULT_PROJECT_ROOT / "models" / MODEL_FILENAME
LOCAL_MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", str(DEFAULT_LOCAL_MODEL)))

# Optional: allow user to tweak classification threshold via env var
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.45))

st.set_page_config(page_title="Tourism Booking Prediction")

st.title("Tourism Booking Prediction")
st.markdown(
    """
Enter the guest and pitch details below and click **Predict** to get the booking prediction and probability.
"""
)

# Build input UI
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

# CACHED loader so model loads only once per process
@st.cache_data(show_spinner=False)
def load_model_cached(local_path: str, hf_repo: str, hf_filename: str, hf_token: str = None):
    """
    Attempt to load model:
      1) from absolute local_path (if present)
      2) from HF single-file download (hf_hub_download) using hf_token if provided
      3) snapshot_download entire repo and search for filename inside snapshot
    Returns loaded model or raises RuntimeError.
    """
    # 1) local
    lp = Path(local_path)
    if lp.exists():
        try:
            model = joblib.load(lp)
            return model
        except Exception as e:
            raise RuntimeError(f"Found local model at {lp} but failed to load: {e}")

    # 2) HF single-file download
    try:
        if hf_token:
            model_file = hf_hub_download(repo_id=hf_repo, filename=hf_filename, token=hf_token)
        else:
            model_file = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
        model = joblib.load(model_file)
        return model
    except Exception as e_hf:
        # 3) fallback: snapshot and search inside
        try:
            snapshot_dir = snapshot_download(repo_id=hf_repo, token=hf_token)
            candidate = Path(snapshot_dir) / hf_filename
            if candidate.exists():
                model = joblib.load(candidate)
                return model
            # maybe the file is nested - try to walk snapshot_dir
            for p in Path(snapshot_dir).rglob(hf_filename):
                try:
                    return joblib.load(p)
                except Exception:
                    continue
            raise RuntimeError(f"HF snapshot found but '{hf_filename}' not present. Snapshot dir: {snapshot_dir}")
        except Exception as final_e:
            # propagate a clear message
            raise RuntimeError(f"Could not load model from local ({lp}) or HF repo ({hf_repo}). Last error: {final_e}") from final_e

def get_hf_token():
    # prefer HUGGINGFACE_HUB_TOKEN; fall back to HF_TOKEN for compatibility
    return os.getenv("HUGGINGFACE_HUB_TOKEN", os.getenv("HF_TOKEN", None))

if st.button("Predict"):
    with st.spinner("Loading model..."):
        try:
            model = load_model_cached(str(LOCAL_MODEL_PATH), MODEL_REPO, MODEL_FILENAME, get_hf_token())
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info(
                "Fixes: upload the model file to the models/ folder, or set LOCAL_MODEL_PATH env var to the absolute path, "
                "or set HUGGINGFACE_HUB_TOKEN for private HF repos. (Don't put tokens directly in the notebook.)"
            )
            st.stop()

    # perform prediction using the model (supports either predict_proba or predict)
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[:, 1][0]
            pred = int(prob >= CLASSIFICATION_THRESHOLD)
            label = "Booked" if pred == 1 else "Not Booked"
            st.subheader("Prediction")
            st.write(f"Prediction: **{label}**")
            st.write(f"Booking probability: **{prob:.3f}** ({prob*100:.1f}%)")
        else:
            pred = model.predict(input_df)[0]
            label = "Booked" if int(pred) == 1 else "Not Booked"
            st.subheader("Prediction")
            st.write(f"Prediction: **{label}**")
    except Exception as pred_e:
        st.error(f"Model loaded but prediction failed: {pred_e}")
