# app.py → LOCAL VERSION (WORKS ON YOUR PC - MOROCCO)

import streamlit as st
import joblib
import re
import spacy
import os

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Mental Alert", layout="centered")

# --- 2. LOCAL PATH ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts_balanced")

# --- 3. LOAD spaCy ---
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_models():
    tfidf_status = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl"))
    svm_model = joblib.load(os.path.join(ARTIFACTS_DIR, "svm_linear.pkl"))
    tfidf_action = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_action.pkl"))
    action_model = joblib.load(os.path.join(ARTIFACTS_DIR, "action_model_5classes.pkl"))
    recommendations = joblib.load(os.path.join(ARTIFACTS_DIR, "recommendations.pkl"))
    return tfidf_status, svm_model, tfidf_action, action_model, recommendations

# --- 5. INITIALIZE ---
nlp = load_nlp()
custom_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
tfidf_status, svm_model, tfidf_action, action_model, recommendations = load_models()

# --- 6. PREPROCESSING ---
def preprocess(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text.lower())
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    return " ".join([t.lemma_.lower() for t in doc 
                     if t.is_alpha and t.lemma_.lower() not in custom_stopwords])

# --- 7. PREDICTION ---
def predict(text: str):
    lemma = preprocess(text)
    status = svm_model.predict(tfidf_status.transform([lemma]))[0]
    enriched = lemma + f" [STATUS: {status}]"
    action = action_model.predict(tfidf_action.transform([enriched]))[0]
    prob = action_model.predict_proba(tfidf_action.transform([enriched]))[0].max()
    rec = recommendations.get(action, "Follow-up recommended")
    return status, action, prob, rec

# --- 8. INTERFACE ---
st.image("img.png", width=100)  # ← YOUR LOCAL IMAGE
st.title("Early Mental Health Alert System")
st.markdown("**Send a message → Get an instant alert**")

user_text = st.text_area("Message:", placeholder="I want to end it...", height=120)

if st.button("Analyze", type="primary"):
    if user_text.strip():
        with st.spinner("Analyzing..."):
            status, action, prob, rec = predict(user_text)
        col1, col2 = st.columns(2)
        with col1: st.metric("Status", status.upper())
        with col2: st.metric("Action", action, f"{prob:.1%}")
        if action == "RISQUE_IMMÉDIAT":
            st.error(f"**RED ALERT**: {rec}")
            if st.button("Call 112"): 
                st.balloons()
                st.success("Simulated call sent!")
        elif action == "RISQUE_ÉLEVÉ": 
            st.warning(rec)
        else: 
            st.success(rec)
    else:
        st.warning("Please enter a message.")
st.caption("Early detection")