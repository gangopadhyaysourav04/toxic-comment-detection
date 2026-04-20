import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import requests
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
from collections import Counter
from bs4 import BeautifulSoup

# --- Configuration & Paths ---
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

def inject_synthetic_data(df):
    neutral = [
        "good morning", "thank you", "how are you", "nice work", "great job", 
        "namaste", "jai hind", "take care", "sab theek hai", "mausam accha hai", 
        "khana kha liya?", "all good bro", "kyu bhai kaisa hai", "badhiya hai",
        "kaise hain aap", "kaise ho", "kaha ho", "kya kar rahe ho", "khana khaya",
        "namaste ji", "aapka swagat hai", "bhout shukriya", "dhanyawad",
        "bahut accha laga", "aap bahut acche hain", "milkar khushi hui",
        "kal milte hain", "dost kaise ho", "family kaisi hai", "sab thik thak",
        "main thik hoon", "aap kaise ho", "kaise ho aap", "kya ho raha hai",
        "kuch nahi", "bas aise hi", "ghar par sab kaise hain", "thik hai bye",
        "ok done", "see you soon", "take care bro", "maze karo"
    ]
    
    # Severe Hinglish toxicity samples (intentional phonetic variety)
    hinglish_toxic = [
        "teri maa chod dunga", "maa ki chut teri", "teri behen ki chut", 
        "gand phad dunga", "gaand me danda", "randi ki aulad", "betichod",
        "madarchod saala", "bhenchod kahi ka", "kutte ka baccha", "haramzada",
        "suar ka bachha", "teri maa ka bhosda", "chal hat bsdk", "bsdk",
        "chutiyapa mat kar", "gandu insaan", "lauda le le mera", "chuchi dikha",
        "porn dekh ke muth mar", "teri maa mere sath hai", "teri aurat meri hai",
        "aap apni g@nd maraye", "aap chutiya hain", "aap apni gand marwa",
        "g@nd marao ji", "lauda khalo aap", "aap bhosadiwale hain",
        "apni aukat me raho chutiye", "maa chuda apni", "behen chuda"
    ]
    
    toxic_intent = [
        "i hope you die", "kill yourself", "go to hell", "i will kill you", 
        "hope your parents die", "get cancer", "die in a hole", "you are worthless",
        "teri maut ho jaye", "mar ja kutte", "zeher kha le", "apni jaan de de",
        "tera khandan khatam ho jaye", "sharm kar besharm", "tu dharti pe bojh hai"
    ]
    
    # Reinforced Hate (Class 0)
    hate = [
        "nigger", "kike", "faggot", "wetback", "porkistanis go back", "indian pigs",
        "mulla", "katwa", "bhakt saale", "librandu", "chamar", "bhangi saale",
        "dalit pigs", "upper caste superiority", "kill all migrants"
    ]

    df_n = pd.DataFrame({"class": 2, "tweet": neutral})
    df_h = pd.DataFrame({"class": 0, "tweet": hate + hinglish_toxic + toxic_intent})
    df_o = pd.DataFrame({"class": 1, "tweet": ["bc", "mc", "saale", "haramkhor", "besharm", "ullu", "kamina"]})
    
    for _df in [df_n, df_h, df_o]:
        _df["clean_tweet"] = _df["tweet"].apply(clean_text)
        
    return pd.concat([df, df_n, df_h, df_o], ignore_index=True)

# ---------------- ADVANCED SCRAPER HUB ----------------

def scrape_reddit(subreddit="india", limit=50):
    url = f"https://www.reddit.com/r/{subreddit}/comments.json?limit={limit}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    }
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    comments = []
    for child in data.get("data", {}).get("children", []):
        body = child.get("data", {}).get("body")
        if body and len(body.split()) > 2:
            comments.append({"tweet": body, "class": 2})
    return comments

def scrape_generic_url(url):
    """
    Extracts text from any public URL using BeautifulSoup.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    }
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Extract textual content from common comment/post containers
    text = soup.get_text(separator=' ')
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    
    return [{"tweet": line, "class": 2} for line in lines[:100]] # Cap at 100 entries

def scrape_github_datasets():
    """
    Fetches raw toxicity datasets from curated GitHub repositories.
    """
    sources = [
        "https://raw.githubusercontent.com/pmathur5k10/Hinglish-Offensive-Text-Classification/master/Hinglish_Profanity_List.csv"
    ]
    records = []
    for url in sources:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                # Basic parsing for profanity lists to convert to synthetic samples
                lines = r.text.splitlines()
                for line in lines:
                    parts = line.split(',')
                    if parts:
                        word = parts[0].strip()
                        records.append({"tweet": f"tum {word} ho", "class": 1}) # Map to Abusive
        except:
            continue
    return records

def update_data_hub(filepath, source_type, identifier=None, uploaded_file=None):
    new_data = []
    
    if source_type == "Reddit" and identifier:
        new_data = scrape_reddit(identifier)
    elif source_type == "Generic URL" and identifier:
        new_data = scrape_generic_url(identifier)
    elif source_type == "GitHub Harvester":
        new_data = scrape_github_datasets()
    elif source_type == "Manual Upload" and uploaded_file is not None:
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
    
    menu = ["Data Collection", "EDA", "Model training", "Testing & Deployment"]
    choice = st.sidebar.selectbox("Pipeline Pipeline", menu, key="main_pipeline_choice")
    
    if "data" not in st.session_state:
        st.session_state.data = load_and_preprocess_data(CSV_PATH)

    if choice == "Data Collection":
        st.header("🌐 Universal Data Harvester")
        st.info("Gather toxicity data from Reddit, GitHub, or any public URL (News/X/Blogs).")
        
        src = st.selectbox("Select Data Source", ["Reddit", "Generic URL", "GitHub Harvester", "Manual Upload"], key="data_source_select")
        
        identifier = ""
        uploaded_file = None
        
        if src == "Reddit":
            identifier = st.text_input("Subreddit Name", "india")
        elif src == "Generic URL":
            identifier = st.text_input("Public URL (e.g. news page, blog thread)", "")
        elif src == "Manual Upload":
            uploaded_file = st.file_uploader("Upload CSV (must contain 'tweet' column)", type="csv")
            
        if st.button("Start Harvesting"):
            with st.spinner("Processing Source..."):
                try:
                    added = update_data_hub(CSV_PATH, src, identifier, uploaded_file)
                    st.session_state.data = load_and_preprocess_data(CSV_PATH)
                    if added > 0:
                        st.success(f"Successfully added **{added}** new records from {src}!")
                    else:
                        st.warning("No new data found. The source might be empty.")
                except Exception as e:
                    if "403" in str(e) or "429" in str(e):
                        st.error("🚨 **Access Blocked!** The website's security blocked our cloud server. This is normal for cloud deployments. Use your **local deployment** to scrape data successfully!")
                    else:
                        st.error(f"Error scraping data: {e}")
                
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
        
        df = inject_synthetic_data(st.session_state.data)
        df = balance_classes(df)
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

    elif choice == "Testing & Deployment":
        st.header("🚀 Deployment & Testing")
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
