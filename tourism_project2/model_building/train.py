import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mlops-training-experiment"))

api = HfApi()

Xtrain_path = os.getenv("XTRAIN_PATH", "hf://datasets/thalaivanan/tourism-project/data/Xtrain.csv")
Xtest_path = os.getenv("XTEST_PATH", "hf://datasets/thalaivanan/tourism-project/data/Xtest.csv")
ytrain_path = os.getenv("YTRAIN_PATH", "hf://datasets/thalaivanan/tourism-project/data/ytrain.csv")
ytest_path = os.getenv("YTEST_PATH", "hf://datasets/thalaivanan/tourism-project/data/ytest.csv")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Ensure y are Series
if isinstance(ytrain, pd.DataFrame) and ytrain.shape[1] == 1:
    ytrain = ytrain.iloc[:, 0]
if isinstance(ytest, pd.DataFrame) and ytest.shape[1] == 1:
    ytest = ytest.iloc[:, 0]

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "NumberOfChildrenVisiting",
]
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

numeric_features = [c for c in numeric_features if c in Xtrain.columns]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

unique_counts = ytrain.value_counts()
if 1 in unique_counts and 0 in unique_counts and unique_counts[1] > 0:
    class_weight = unique_counts[0] / unique_counts[1]
else:
    class_weight = 1

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    remainder="drop",
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, use_label_encoder=False, eval_metric="logloss", random_state=42)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", float(mean_score))
            mlflow.log_metric("std_test_score", float(std_score))

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    classification_threshold = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.45))

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": float(train_report.get("accuracy", 0)),
        "train_precision": float(train_report.get("1", {}).get("precision", 0)),
        "train_recall": float(train_report.get("1", {}).get("recall", 0)),
        "train_f1-score": float(train_report.get("1", {}).get("f1-score", 0)),
        "test_accuracy": float(test_report.get("accuracy", 0)),
        "test_precision": float(test_report.get("1", {}).get("precision", 0)),
        "test_recall": float(test_report.get("1", {}).get("recall", 0)),
        "test_f1-score": float(test_report.get("1", {}).get("f1-score", 0)),
    })

    model_path = os.getenv("MODEL_OUTPUT_PATH", "best_tourism_model_v1.joblib")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    repo_id = os.getenv("HF_MODEL_REPO", "thalaivanan/tourism_model")
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        try:
            create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        except HfHubHTTPError:
            pass

    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            repo_type="model",
        )
    except Exception:
        pass

print("Model training and upload completed.")
