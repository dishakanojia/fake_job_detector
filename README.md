# 🕵️ Fake Job Posting Detector

Flags fraudulent job ads for manual review, using NLP (TF-IDF) and classic ML. Built on 17,880 real postings, of which only 4.8% are fake.

**Results on 3,214 held-out postings (144 fake)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| "Always say real" baseline | 95.5% | 0 | 0 | 0 |
| Linear SVM, default threshold | 98.4% | 0.854 | 0.771 | 0.810 |
| **Linear SVM, tuned threshold** (recall ≥ 85% rule) | 98.0% | 0.752 | **0.840** | 0.793 |

The baseline row is the point: on imbalanced data, accuracy hides everything.

## What's inside
| File | What it is |
|---|---|
| [`fake_job_detector.ipynb`](fake_job_detector.ipynb) | Full walkthrough with outputs: EDA → cleaning → NLP → 7 models → tuning → threshold → clustering |
| [`features.py`](features.py) | Text-cleaning pipeline + engineered features, shared by the notebook and the app |
| [`app.py`](app.py) | Streamlit app: paste a posting → verdict + the words that drove it |

## Approach
1. **EDA.** Missing info is a signal: no company profile → 17.7% fake vs 1.9%. No logo → 15.9% vs 2.0%.
2. **Cleaning.** Removed 1,814 duplicate postings so copies can't leak from train into test.
3. **NLP pipeline.** Lowercase → strip URLs/HTML/emojis/punctuation → stopwords → Porter stemming → **TF-IDF with 1–2-grams** (20k features).
4. **Feature engineering.** 8 numeric features (has_company_profile, has_logo, log word count…) next to the text.
5. **Bag of Words vs TF-IDF**, compared with stratified 5-fold CV.
6. **Model race.** Logistic Regression, KNN, Naive Bayes, Decision Tree, Linear SVM, Random Forest, XGBoost, plus stacking. Class weights handle the imbalance.
7. **Tuning.** GridSearchCV over the *whole* pipeline (vectorizer + model, leak-free) and RandomizedSearchCV for Logistic Regression.
8. **Threshold tuning** on out-of-fold predictions: catch ≥ 85% of scams with the fewest false alarms.
9. **Explainability.** Linear SVM weights show the scam vocabulary ("apply using link", "earn", "data entry", "clerk", offshore/subsea oil jobs).
10. **Unsupervised.** TruncatedSVD (LSA) + K-Means found job-topic clusters. One generic customer-service/admin cluster holds a 13.6% fraud rate vs 0.9% for web development.

## Run it
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
# download the dataset (50 MB) into data/
curl -L --create-dirs -o data/fake_job_postings.csv https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv
jupyter notebook fake_job_detector.ipynb   # run all cells (~20 min on 4 cores) -> creates model.joblib
streamlit run app.py
```

Data: the EMSCAD *Employment Scam Aegean Dataset* (University of the Aegean), also on [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).

## Resume bullet
> Built an NLP fraud detector for job postings (17.9k ads, 4.8% fraud) using a text-cleaning pipeline, TF-IDF n-grams and engineered "missing-info" features. Compared 7 classifiers + stacking with stratified CV and tuned a leak-free Linear SVM pipeline with GridSearchCV. **Catches 84% of scams on held-out data at 75% precision** (vs 0% for an accuracy-maximising baseline). Explainable Streamlit app highlights the words behind each flag.
