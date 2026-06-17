from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/xgb_underwriter_pipeline.joblib")
METRICS_PATH = Path("models/model_metrics.joblib")
DATA_PATH = Path("data/expenses.csv")

st.set_page_config(
    page_title="Glass Box Underwriter Demo",
    page_icon="🔍",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).drop_duplicates()

@st.cache_data
def load_metrics():
    return joblib.load(METRICS_PATH)

model = load_model()
df = load_data()
metrics = load_metrics()

st.title("The Glass Box Underwriter")
st.subheader("A lightweight What-If demo for predictive underwriting")

st.write(
    "Change the applicant inputs below and see how the model's risk prediction changes. "
    "This recreates the spirit of Google's What-If Tool using the same kind of interactive testing, "
    "but with a Streamlit app that can be hosted and linked from the presentation."
)

with st.expander("Important framing for reviewers", expanded=True):
    st.write(
        "This is a classroom demonstration, not a real underwriting system. "
        "The model predicts whether annual medical charges are above $10,000, which is used here as a proxy for Bad Risk. "
        "A real insurer would need richer consented data, formal fairness testing, legal review, human oversight, "
        "and an appeals process before using a model like this."
    )

left, right = st.columns([1, 1])

with left:
    st.header("Applicant inputs")

    age = st.slider("Age", min_value=18, max_value=64, value=35)
    bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=30.0, step=0.1)
    children = st.slider("Children", min_value=0, max_value=5, value=0)
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    applicant = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    st.dataframe(applicant, use_container_width=True)

proba_bad = float(model.predict_proba(applicant)[0, 1])
pred_bad = int(proba_bad >= 0.50)
label = "Bad Risk" if pred_bad else "Good Risk"

# Build a baseline comparison using the user's current inputs, but non-smoker.
baseline = applicant.copy()
baseline["smoker"] = "no"
baseline_proba_bad = float(model.predict_proba(baseline)[0, 1])
delta = proba_bad - baseline_proba_bad

with right:
    st.header("Model output")

    st.metric("Prediction", label)
    st.metric("Probability of Bad Risk", f"{proba_bad:.1%}")
    st.metric("Change vs non-smoker baseline", f"{delta:+.1%}")

    st.progress(proba_bad)

    if pred_bad:
        st.warning("The model would flag this applicant as higher-cost under the $10,000 threshold.")
    else:
        st.success("The model would flag this applicant as lower-cost under the $10,000 threshold.")

    st.caption("Default decision threshold: 50% probability of Bad Risk.")

st.divider()

st.header("Try these what-if questions")
q1, q2, q3 = st.columns(3)

with q1:
    st.write("**Smoking sensitivity**")
    st.write("Change smoker from no to yes. This should move the prediction the most because smoking is the dominant signal in the project.")

with q2:
    st.write("**BMI and age sensitivity**")
    st.write("Increase BMI or age while keeping smoker fixed. The prediction should usually move more gradually.")

with q3:
    st.write("**Fairness/proxy check**")
    st.write("Change sex or region and watch whether the prediction changes. These are the fields a reviewer should audit carefully.")

st.divider()

st.header("Model card snapshot")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Precision", f"{metrics['precision']:.1%}")
m3.metric("Recall", f"{metrics['recall']:.1%}")
m4.metric("F1", f"{metrics['f1']:.3f}")
m5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.write(
    "The model was trained on the Kaggle medical cost dataset used in the presentation. "
    "The binary target is `Bad Risk = charges > $10,000`."
)

st.divider()

st.header("Optional audit table")
sample = df.sample(10, random_state=42).copy()
sample["bad_risk_actual"] = (sample["charges"] > 10000).astype(int)
sample["bad_risk_probability"] = model.predict_proba(sample[metrics["features"]])[:, 1]
sample["model_prediction"] = sample["bad_risk_probability"].apply(lambda p: "Bad Risk" if p >= 0.5 else "Good Risk")

st.dataframe(
    sample[["age", "sex", "bmi", "children", "smoker", "region", "charges", "bad_risk_actual", "bad_risk_probability", "model_prediction"]],
    use_container_width=True,
)