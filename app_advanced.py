import streamlit as st
import feedparser
import joblib
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from utils import preprocess_text

# --- Load Models & Assets ---
@st.cache_resource
def load_assets():
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        lr_model = joblib.load('lr_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
    except:
        lr_model, tfidf = None, None
    return bert_model, lr_model, tfidf

bert_model, lr_model, tfidf = load_assets()

# --- Functions ---
# def get_live_news():
#     rss_url = "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US&gl=US"
#     feed = feedparser.parse(rss_url)
#     return [entry.title for entry in feed.entries[:15]]


import urllib.parse  # <--- Add this import at the top of your file


def get_dynamic_context(user_query):
    # Remove common words to get better search results (e.g., "is", "the", "of")
    query_words = [word for word in user_query.split() if word.lower() not in ["is", "the", "of", "a", "an", "in"]]
    search_keywords = " ".join(query_words[:6]) # Use top 6 meaningful words
    
    encoded_keywords = urllib.parse.quote(search_keywords)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keywords}&hl=en-US"
    
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:10]]

# def hybrid_prediction(news_text):
#     # Part A: Statistical ML Prediction
#     if lr_model and tfidf:
#         # Preprocess text using your shared utils function
#         clean_text = preprocess_text(news_text)
#         vectorized_text = tfidf.transform([clean_text])
#         ml_score = lr_model.predict_proba(vectorized_text)[0][1] 
#     else:
#         ml_score = 0.5

#     # Part B: Semantic BERT Check (Comparing input text with live headlines)
#     live_headlines = get_live_news()
#     input_embedding = bert_model.encode(news_text, convert_to_tensor=True)
#     trusted_embeddings = bert_model.encode(live_headlines, convert_to_tensor=True)
    
#     similarity_scores = util.cos_sim(input_embedding, trusted_embeddings)
#     max_sim_score = float(similarity_scores.max())
    
#     # Final Hybrid Score
#     final_score = (ml_score * 0.4) + (max_sim_score * 0.6)
#     return final_score, max_sim_score, ml_score




# def hybrid_prediction(news_text):
#     # 1. Statistical ML Prediction (Needs Cleaning)
#     if lr_model and tfidf:
#         clean_text = preprocess_text(news_text) # Shared cleaning logic
#         vectorized_text = tfidf.transform([clean_text])
#         # Probability that the news is REAL
#         ml_score = lr_model.predict_proba(vectorized_text)[0][1] 
#     else:
#         ml_score = 0.5

#     # 2. Semantic BERT Check (Needs Natural Text)
#     live_headlines = get_live_news()
#     # BERT understands context better WITHOUT heavy cleaning
#     input_embedding = bert_model.encode(news_text, convert_to_tensor=True)
#     trusted_embeddings = bert_model.encode(live_headlines, convert_to_tensor=True)
    
#     similarity_scores = util.cos_sim(input_embedding, trusted_embeddings)
#     max_sim_score = float(similarity_scores.max())
    
#     # 3. SMART HYBRID LOGIC
#     # If a high similarity is found in live news, it's almost certainly REAL.
#     if max_sim_score > 0.75:
#         final_score = max_sim_score # Trust the live match
#     else:
#         # If no live match, rely more on the ML model's pattern detection
#         final_score = (ml_score * 0.7) + (max_sim_score * 0.3)
        
#     return final_score, max_sim_score, ml_score





# def hybrid_prediction(news_text):
#     # --- PATH 1: Statistical ML (CLEANED) ---
#     if lr_model and tfidf:
#         # We clean the text ONLY for the Logistic Regression model
#         cleaned_for_ml = preprocess_text(news_text) 
#         vectorized_text = tfidf.transform([cleaned_for_ml])
#         ml_score = lr_model.predict_proba(vectorized_text)[0][1] 
#     else:
#         ml_score = 0.5

#     # --- PATH 2: Semantic BERT (NATURAL) ---
#     live_headlines = get_live_news()
    
#     # We use the raw 'news_text' so BERT sees the full grammar/context
#     input_embedding = bert_model.encode(news_text, convert_to_tensor=True)
#     trusted_embeddings = bert_model.encode(live_headlines, convert_to_tensor=True)
    
#     similarity_scores = util.cos_sim(input_embedding, trusted_embeddings)
#     max_sim_score = float(similarity_scores.max())
    
#     # --- PATH 3: Hybrid Logic ---
#     # Weighting: 40% ML, 60% BERT
#     final_score = (ml_score * 0.4) + (max_sim_score * 0.6)
    
#     return final_score, max_sim_score, ml_score





def hybrid_prediction(news_text):
    # --- PATH 1: Statistical ML (Cleaned) ---
    if lr_model and tfidf:
        cleaned_for_ml = preprocess_text(news_text) 
        vectorized_text = tfidf.transform([cleaned_for_ml])
        ml_score = lr_model.predict_proba(vectorized_text)[0][1] 
    else:
        ml_score = 0.5

    # --- PATH 2: Semantic BERT (Natural) ---
    live_headlines = get_dynamic_context(news_text)
    
    # SAFETY CHECK: If no news matches are found, we cannot use BERT
    if not live_headlines:
        # If no live news confirms it, we trust the ML pattern.
        # If ML score is high (>0.7), we stay cautious (0.5).
        # If ML score is low (<0.5), it's likely fake.
        final_score = ml_score if ml_score < 0.5 else 0.45
        return final_score, 0.0, ml_score

    # If we have headlines, proceed with BERT
    input_embedding = bert_model.encode(news_text, convert_to_tensor=True)
    trusted_embeddings = bert_model.encode(live_headlines, convert_to_tensor=True)
    
    similarity_scores = util.cos_sim(input_embedding, trusted_embeddings)
    max_sim_score = float(similarity_scores.max())
    
    # --- PATH 3: SMART HYBRID LOGIC ---
    if max_sim_score > 0.50:
        final_score = (max_sim_score * 0.9) + (ml_score * 0.1)
    elif max_sim_score < 0.15:
        final_score = (ml_score * 0.7) + (max_sim_score * 0.3)
    else:
        final_score = (ml_score * 0.5) + (max_sim_score * 0.5)
    
    return final_score, max_sim_score, ml_score

# --- Streamlit UI ---
st.set_page_config(page_title="AI Fact Checker Pro", layout="wide")
st.title("🛡️ Hybrid News Verifier v2.0")

# CHANGED: Use text_area instead of text_input for URLs
user_news = st.text_area("Paste News Content or Headline here:", height=200)

if st.button("Check Authenticity"):
    if user_news.strip():
        with st.spinner("Analyzing content..."):
            # 1. Hybrid Analysis
            final, sim, ml = hybrid_prediction(user_news)
            
            # 2. Layout: Results & Chart
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Analysis Results")
                st.metric("Final Veracity Score", f"{final*100:.1f}%")
                
                if final > 0.65:
                    st.success("Verdict: HIGHLY LIKELY REAL")
                elif final > 0.45:
                    st.warning("Verdict: UNCERTAIN / MIXED")
                else:
                    st.error("Verdict: HIGHLY LIKELY FAKE")

            with col2:
                plot_data = pd.DataFrame({
                    "Method": ["Statistical (ML)", "Semantic (BERT)", "Hybrid Result"],
                    "Confidence": [ml, sim, final]
                })
                fig = px.bar(plot_data, x="Method", y="Confidence", color="Method", 
                             title="Prediction Breakdown", range_y=[0, 1])
                st.plotly_chart(fig)
                
            with st.expander("Live Sources Checked Against"):
                sources = get_dynamic_context(user_news)
                if sources:
                    st.write(sources)
                else:
                    st.info("No live news matches found for this specific query.")
    else:
        st.error("Please paste some text to analyze.")