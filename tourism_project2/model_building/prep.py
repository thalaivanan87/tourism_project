# for data manipulation
import pandas as pd
import sklearn  # kept to match reference notebook style
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting categorical data into numerical representation
from sklearn.preprocessing import LabelEncoder
# for optional Hugging Face dataset upload
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/thalaivanan/tourism-project/tourism-project.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

print("Loaded dataset shape:", df.shape)

# Drop common identifier columns if present
id_candidates = ["Unnamed: 0", "CustomerID", "ID", "Id", "id", "booking_id", "reservation_id"]
for c in id_candidates:
    if c in df.columns:
        print(f"Dropping identifier column: {c}")
        df.drop(columns=[c], inplace=True, errors="ignore")

# Use LabelEncoder on object/category columns
label_encoder = LabelEncoder()
obj_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
print("Categorical/object columns to encode:", obj_cols)
for col in obj_cols:
    df[col] = df[col].fillna("__MISSING__").astype(str)
    df[col] = label_encoder.fit_transform(df[col])
    print(f"Encoded column: {col}")

TARGET_COL = "ProdTaken"
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset. Columns: {list(df.columns)}")
print("Using target column:", TARGET_COL)

# Separate features and label
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Stratify when sensible (classification-like target)
stratify = y if y.nunique() > 1 else None

# Train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=stratify
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="thalaivanan/tourism-project",
        repo_type="dataset",
    )
