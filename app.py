import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from collections import Counter# --- Configuration & Paths ---
st.set_page_config(
    page_title="Toxic Comment Detection Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSV_PATH = "labeled_data_with_more_hinglish_final_v2_cleaned.csv"
MODEL_PATH = "best_model.pkl"
VEC_PATH = "vectorizer.pkl"

# --- Styling ---
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #6366f1; color: white; font-weight: bold; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- PREPROCESSING LOGIC ----------------

def clean_text(text):
    text = str(text).lower()
    # Comprehensive mappings for Hinglish phonetic variations + roots
    mappings = {
        "g@ndu": "gandu", "g@ndoo": "gandu", "g4ndu": "gandu", "gaand": "gandu", "g@nd": "gaand", "gand": "gaand",
        "chutiyaa": "chutiya", "chut!ya": "chutiya", "chuity": "chutiya",
        "madarch0d": "madarchod", "mc": "madarchod", "bc": "bhenchod",
        "bhench0d": "bhenchod", "behenchod": "bhenchod", "bhenchod": "bhenchod",
        "lawde": "lauda", "lodu": "lauda", "lowde": "lauda", "lodhu": "lauda",
        "kutter ka baccha": "kutta", "kutiya": "kutta",
        "bhen ke lode": "bhenchod", "maa ki chut": "maachod", "maachod": "maachod",
        "chod": "chodo", "chuda": "chodo", "chud": "chodo", "chodo": "chodo", "maraye": "marwa", "mara": "marwa",
        "randi": "randi", "raand": "randi", "chinaal": "randi", "bhadva": "bhadva"
    }
    for k, v in mappings.items():
        text = text.replace(k, v)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame(columns=["tweet", "class", "clean_tweet"])
    
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df.dropna(subset=["class", "tweet"])
    df["class"] = df["class"].astype(int)
    df = df[df["class"].isin([0, 1, 2])]
    df = df.drop_duplicates(subset=["tweet"])
    df["clean_tweet"] = df["tweet"].apply(clean_text)
    return df


def update_data_hub(filepath, uploaded_file=None):
    new_data = []
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "tweet" in df_upload.columns:
                if "class" not in df_upload.columns: df_upload["class"] = 2
                new_data = df_upload[["tweet", "class"]].to_dict('records')
        except:
            pass

    if not new_data: return 0
    df_new = pd.DataFrame(new_data)
    try:
        df_old = pd.read_csv(filepath)
        df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["tweet"])
        added = len(df_combined) - len(df_old)
        df_combined.to_csv(filepath, index=False)
        return added
    except:
        df_new.to_csv(filepath, index=False)
        return len(df_new)

# ---------------- MODELING LOGIC ----------------

def balance_classes(df):
    counts = df["class"].value_counts()
    # Cap samples to speed up training while keeping it balanced
    max_n = min(counts.max(), 10000) 
    return pd.concat([
        resample(df[df["class"] == c], replace=True, n_samples=max_n, random_state=42)
        for c in df["class"].unique()
    ]).reset_index(drop=True)

def train_pipeline(df):
    # CHARACTER-LEVEL VECTORIZER: Crucial for phonetic variations in Hinglish
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        analyzer='char_wb', # Character-level N-grams within word boundaries
        ngram_range=(2, 5), 
        min_df=1, # Be as sensitive as possible to rare toxic roots
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["clean_tweet"])
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    from sklearn.linear_model import SGDClassifier
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "High-Speed SGD": SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(alpha=0.1)
    }
    
    best_m, best_name, best_acc = None, "", 0
    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"acc": acc, "cm": confusion_matrix(y_test, preds)}
        if acc > best_acc:
            best_acc, best_m, best_name = acc, clf, name
            
    return best_m, best_name, results, vectorizer

# ---------------- UI DASHBOARD ----------------

def main():
    st.title("🛡️ Toxic Comment Detection Pipeline")
    
    menu = ["Data Collection", "EDA", "Model training", "Testing"]
    choice = st.sidebar.selectbox("Pipeline Pipeline", menu, key="main_pipeline_choice")
    
    if "data" not in st.session_state:
        st.session_state.data = load_and_preprocess_data(CSV_PATH)

    if choice == "Data Collection":
        st.header("📂 Data Collection")
        st.info("Upload new CSV datasets to expand the training data.")
        
        uploaded_file = st.file_uploader("Upload CSV (must contain 'tweet' column)", type="csv")
            
        if st.button("Start Processing"):
            with st.spinner("Processing file..."):
                try:
                    added = update_data_hub(CSV_PATH, uploaded_file)
                    st.session_state.data = load_and_preprocess_data(CSV_PATH)
                    if added > 0:
                        st.success(f"Successfully added **{added}** new records!")
                    else:
                        st.warning("No new data found. The file might be empty or improperly formatted.")
                except Exception as e:
                    st.error(f"Error processing data: {e}")
                
        st.write(f"Total entries in dataset: **{len(st.session_state.data)}**")
        st.dataframe(st.session_state.data.head(15), use_container_width=True)

    elif choice == "EDA":
        st.header("📊 Exploratory Data Analysis")
        df = st.session_state.data
        if df.empty: st.warning("No data found."); return
        
        col1, col2 = st.columns(2)
        with col1:
            counts = df["class"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            counts["Label"] = counts["Class"].map({0:"Hate", 1:"Offensive", 2:"Non-toxic"})
            st.plotly_chart(px.bar(counts, x="Label", y="Count", color="Label", title="Class Distribution"), use_container_width=True)
        
        with col2:
            df["len"] = df["clean_tweet"].apply(lambda x: len(str(x).split()))
            st.plotly_chart(px.histogram(df, x="len", color="class", barmode='overlay', title="Message Length Distribution"), use_container_width=True)
            
        st.subheader("Word Frequency")
        c_id = st.radio("Select Class", [0, 1, 2], format_func=lambda x: {0:"Hate",1:"Offensive",2:"Clean"}[x], horizontal=True)
        words = " ".join(df[df["class"]==c_id]["clean_tweet"]).split()
        top = pd.DataFrame(Counter(words).most_common(20), columns=["Word", "Freq"])
        st.plotly_chart(px.bar(top, x="Freq", y="Word", orientation='h', title=f"Top Words in Class {c_id}"), use_container_width=True)

    elif choice == "Model training":
        st.header("⚙️ Model Selection & Training")
        if st.session_state.data.empty: st.warning("No data found."); return
        
        df = balance_classes(st.session_state.data)
        st.write(f"Training on {len(df)} samples after balancing.")
        
        if st.button("Train Pipeline"):
            with st.spinner("Finding best model..."):
                model, name, results, vec = train_pipeline(df)
                joblib.dump(model, MODEL_PATH)
                joblib.dump(vec, VEC_PATH)
                st.success(f"Best Model: {name}")
                st.table(pd.DataFrame({m: [v['acc']] for m,v in results.items()}, index=["Accuracy"]).T)
                
                fig, ax = plt.subplots()
                sns.heatmap(results[name]['cm'], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["Hate", "Off", "Clean"], yticklabels=["Hate", "Off", "Clean"])
                st.pyplot(fig)

    elif choice == "Testing":
        st.header("🚀 Testing")
        if not os.path.exists(MODEL_PATH): st.error("No model found. Train first."); return
        
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VEC_PATH)
        
        text = st.text_area("Input Text (Hinglish/English)")
        if st.button("Predict"):
            if text:
                cleaned = clean_text(text)
                X_in = vec.transform([cleaned])
                pred = model.predict(X_in)[0]
                prob = model.predict_proba(X_in)[0][pred]
                
                # Simplified Labels
                labels = {
                    0: "Hate Speech ❌", 
                    1: "Abusive/Offensive ⚠️", 
                    2: "Non-toxic ✅"
                }
                st.subheader(f"Result: {labels[pred]}")
                st.progress(prob)
                st.write(f"Confidence: {prob*100:.2f}%")
            else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
