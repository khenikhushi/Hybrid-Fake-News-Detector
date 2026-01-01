import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from utils import preprocess_text
import urllib.parse
import feedparser
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. Load Everything
print("Loading models...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
lr_model = joblib.load('lr_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.joblib')

def get_dynamic_context(user_query):
    query_words = [word for word in user_query.split() if word.lower() not in ["is", "the", "of", "a", "an", "in"]]
    search_keywords = " ".join(query_words[:6])
    encoded_keywords = urllib.parse.quote(search_keywords)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keywords}&hl=en-US"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:10]]

def hybrid_prediction(news_text):
    # ML Path
    cleaned_for_ml = preprocess_text(news_text) 
    vectorized_text = tfidf.transform([cleaned_for_ml])
    ml_score = lr_model.predict_proba(vectorized_text)[0][1] 

    # BERT Path
    live_headlines = get_dynamic_context(news_text)
    if not live_headlines:
        max_sim_score = 0.0
    else:
        input_embedding = bert_model.encode(news_text, convert_to_tensor=True)
        trusted_embeddings = bert_model.encode(live_headlines, convert_to_tensor=True)
        similarity_scores = util.cos_sim(input_embedding, trusted_embeddings)
        max_sim_score = float(similarity_scores.max())
    
    # Hybrid Logic (Matches your app_advanced.py)
    if max_sim_score > 0.50:
        final_score = (max_sim_score * 0.9) + (ml_score * 0.1)
    elif max_sim_score < 0.15:
        final_score = (ml_score * 0.7) + (max_sim_score * 0.3)
    else:
        final_score = (ml_score * 0.5) + (max_sim_score * 0.5)
    
    # Convert score to label: >0.5 is 'real', <=0.5 is 'fake'
    pred_label = "real" if final_score > 0.5 else "fake"
    return pred_label, final_score


# 2. Load Dataset and Test
df = pd.read_csv('fake_news_dataset.csv').sample(50) # Testing on 50 samples
print(f"Testing Hybrid Model on {len(df)} samples...")

predictions = []
prob_scores = []

# Convert labels to binary (IMPORTANT)
true_labels = df['label'].apply(
    lambda x: 1 if str(x).lower() in ['real', '1', 'true'] else 0
).tolist()

for index, row in df.iterrows():
    pred_label, score = hybrid_prediction(row['text'])
    
    predictions.append(1 if pred_label == "real" else 0)
    prob_scores.append(score)

    print(f"Processed {len(predictions)}/50...")

# 3. Final Calculation
total_accuracy = accuracy_score(true_labels, predictions)


# ===============================
# IIT-LEVEL EVALUATION METRICS
# ===============================

precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
roc_auc = roc_auc_score(true_labels, prob_scores)

cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()

# False Positive Reduction Rate
# (Compared to a random baseline)
fpr = fp / (fp + tn)
false_positive_reduction = 1 - fpr

print("\n" + "="*40)
print("HYBRID MODEL – DETAILED EVALUATION")
print("="*40)

print(f"Accuracy   : {total_accuracy:.2%}")
print(f"Precision  : {precision:.2%}")
print(f"Recall     : {recall:.2%}")
print(f"F1-Score   : {f1:.2%}")
print(f"ROC-AUC    : {roc_auc:.2%}")

print("\nConfusion Matrix:")
print(cm)

print("\nFalse Positive Reduction Rate:")
print(f"{false_positive_reduction:.2%}")

print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=["Fake", "Real"]))

print("="*40)
