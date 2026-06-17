from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Install xgboost first: pip install xgboost") from exc

RNG = 42
DATA_PATH = Path("data/expenses.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop_duplicates().copy()

# Matches the notebook and slide deck: Bad Risk = annual charges above $10,000
df["bad_risk"] = (df["charges"] > 10000).astype(int)

# Version A: exact notebook model, includes all six original features.
# Version B: for a stricter pricing demo, remove sex and region from FEATURES.
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
NUM_COLS = ["age", "bmi", "children"]
CAT_COLS = ["sex", "smoker", "region"]

X = df[FEATURES]
y = df["bad_risk"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])

model = XGBClassifier(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RNG,
)

pipe = Pipeline([
    ("pre", preprocess),
    ("clf", model),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RNG, stratify=y
)

pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "features": FEATURES,
    "threshold": 10000,
}

joblib.dump(pipe, MODEL_DIR / "xgb_underwriter_pipeline.joblib")
joblib.dump(metrics, MODEL_DIR / "model_metrics.joblib")

print("Saved model to models/xgb_underwriter_pipeline.joblib")
print("Saved metrics to models/model_metrics.joblib")
print(metrics)