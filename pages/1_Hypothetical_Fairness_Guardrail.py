import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Hypothetical Fairness Guardrail",
    layout="wide"
)

st.title("Hypothetical Fairness Guardrail")
st.caption("A Responsible AI guardrail demo for predictive underwriting.")

st.warning(
    "This page is hypothetical. The original dataset does not include race or ZIP code. "
    "The demo dataset uses synthetic race and ZIP code values only to show how a guardrail could work with richer data."
)

dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    ["Original expenses.csv", "Synthetic guardrail demo data"]
)

if dataset_choice == "Original expenses.csv":
    data_path = DATA_DIR / "expenses.csv"
else:
    data_path = DATA_DIR / "expenses_guardrail_demo.csv"

df = pd.read_csv(data_path).drop_duplicates()

required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(), width="stretch")

df["good_risk"] = np.where(df["charges"] <= 10000, 1, 0)

st.info(
    "Target definition: charges at or below $10,000 = Good Risk. "
    "Charges above $10,000 = Bad Risk / flag for review."
)

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 40, 55, 100],
    labels=["18-25", "26-40", "41-55", "56+"]
)

df["bmi_group"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

df["children_group"] = df["children"].apply(
    lambda x: "0 children" if x == 0 else "1-2 children" if x <= 2 else "3+ children"
)

if "zip_code" in df.columns:
    df["zip_prefix"] = df["zip_code"].astype(str).str[:3]

if "zipcode" in df.columns:
    df["zip_prefix"] = df["zipcode"].astype(str).str[:3]

available_fairness_columns = [
    col for col in [
        "sex",
        "region",
        "smoker",
        "age_group",
        "bmi_group",
        "children_group",
        "race",
        "zipcode",
        "zip_code",
        "zip_prefix"
    ]
    if col in df.columns
]

st.sidebar.header("Guardrail Settings")

fairness_column = st.sidebar.selectbox(
    "Compare model outcomes by:",
    available_fairness_columns
)

threshold = st.sidebar.slider(
    "Max allowed approval gap",
    min_value=1,
    max_value=30,
    value=10,
    help="If the gap between groups exceeds this percentage, the guardrail triggers an alert."
)

features = ["age", "sex", "bmi", "children", "smoker", "region"]
target = "good_risk"

X = df[features]
y = df[target]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy * 100:.2f}%")

fairness_df = X_test.copy()
fairness_df["actual_good_risk"] = y_test.values
fairness_df["predicted_good_risk"] = predictions
fairness_df[fairness_column] = df.loc[fairness_df.index, fairness_column]

fairness_summary = fairness_df.groupby(fairness_column).agg(
    total_applicants=("predicted_good_risk", "count"),
    predicted_good_risk_rate=("predicted_good_risk", "mean"),
    actual_good_risk_rate=("actual_good_risk", "mean")
).reset_index()

fairness_summary["predicted_good_risk_rate_pct"] = (
    fairness_summary["predicted_good_risk_rate"] * 100
).round(2)

fairness_summary["actual_good_risk_rate_pct"] = (
    fairness_summary["actual_good_risk_rate"] * 100
).round(2)

max_gap = (
    fairness_summary["predicted_good_risk_rate_pct"].max()
    - fairness_summary["predicted_good_risk_rate_pct"].min()
)

st.subheader("Fairness Results")

st.dataframe(
    fairness_summary[
        [
            fairness_column,
            "total_applicants",
            "predicted_good_risk_rate_pct",
            "actual_good_risk_rate_pct"
        ]
    ].rename(
        columns={
            "total_applicants": "Total Applicants",
            "predicted_good_risk_rate_pct": "Predicted Good Risk Rate (%)",
            "actual_good_risk_rate_pct": "Actual Good Risk Rate (%)"
        }
    ),
    use_container_width=True
)

st.metric("Largest Approval Gap", f"{max_gap:.2f} percentage points")

if max_gap > threshold:
    st.error(
        f"GUARDRAIL ALERT: The approval gap is {max_gap:.2f} percentage points, "
        f"which is above the {threshold}% threshold."
    )
    st.write("Recommendation: Pause deployment and investigate possible disparate impact.")
else:
    st.success(
        f"GUARDRAIL PASSED: The approval gap is {max_gap:.2f} percentage points, "
        f"which is within the {threshold}% threshold."
    )
    st.write("Recommendation: Continue monitoring fairness before deployment.")

st.subheader("Plain-English Interpretation")
st.write(
    f"The app compared predicted Good Risk rates across groups in `{fairness_column}`. "
    f"The largest difference between groups was {max_gap:.2f} percentage points."
)

st.subheader("Responsible AI Notes")
st.write(
    """
This guardrail does not prove the model is fair. It is an early warning system.

A full fairness review should also include:

- Protected attributes such as race, sex, age, and disability status
- Proxy variables such as ZIP code, region, income, occupation, and education
- False positive and false negative rates across groups
- Appeal outcomes and denial reasons
- Ongoing monitoring after deployment
- Human review for borderline or high-impact decisions
"""
)