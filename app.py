"""Streamlit app: paste a job posting -> fraud risk + the words that drove it.

Run:  streamlit run app.py   (after running the notebook once to create model.joblib)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import make_features

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")


@st.cache_resource
def load_artifact():
    return joblib.load(Path(__file__).parent / "model.joblib")


art = load_artifact()
model, threshold = art["model"], art["threshold"]

st.title("🕵️ Fake Job Posting Detector")
st.caption(f"Linear SVM + TF-IDF · test recall {art['test_metrics']['recall']:.0%}, "
           f"precision {art['test_metrics']['precision']:.0%}")

title = st.text_input("Job title", "Data Entry Clerk - Work From Home")
company_profile = st.text_area("Company profile (leave empty if the ad has none)", "")
description = st.text_area("Job description", height=180, value=(
    "Earn $500 per day from home! No experience needed. We are hiring immediately. "
    "Just send your bank details and a small registration fee to get started today."))
requirements = st.text_area("Requirements", "Must have a computer and internet.")
benefits = st.text_area("Benefits", "")
c1, c2, c3, c4 = st.columns(4)
has_logo = c1.checkbox("Has company logo")
has_questions = c2.checkbox("Has screening questions")
telecommuting = c3.checkbox("Remote job", value=True)
has_salary = c4.checkbox("Salary listed")

if st.button("Check posting", type="primary"):
    raw = pd.DataFrame([{
        "title": title, "company_profile": company_profile, "description": description,
        "requirements": requirements, "benefits": benefits, "salary_range": "yes" if has_salary else "",
        "has_company_logo": int(has_logo), "has_questions": int(has_questions), "telecommuting": int(telecommuting),
    }])
    X = make_features(raw)
    score = float(model.decision_function(X)[0])

    if score >= threshold:
        st.error(f"⚠️ Likely FAKE. Send to manual review (score {score:.2f} ≥ threshold {threshold:.2f})")
    else:
        st.success(f"✅ Looks legitimate (score {score:.2f} < threshold {threshold:.2f})")

    # explain: contribution of each feature = its value x its SVM weight
    prep, svm = model.named_steps["prep"], model.named_steps["model"]
    values = prep.transform(X).toarray()[0]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    contrib = pd.Series(values * svm.coef_[0], index=names)
    contrib = contrib[contrib != 0].sort_values()
    st.subheader("Why?")
    left, right = st.columns(2)
    left.write("🚩 **Pushing towards fake**")
    left.dataframe(contrib[contrib > 0][::-1].head(8).round(3).rename("impact"))
    right.write("🟢 **Pushing towards real**")
    right.dataframe(contrib[contrib < 0].head(8).round(3).rename("impact"))
