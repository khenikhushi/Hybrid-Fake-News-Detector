# Fake News Detection System using Machine Learning & NLP
import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import your custom preprocessing from utils.py
from utils import preprocess_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------
# 1. Setup & Resources
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -----------------------------
# 2. Load & Clean Dataset
# -----------------------------
try:
    # UPDATED: Changed filename to match your uploaded dataset
    df = pd.read_csv("fake_news_dataset.csv")
    
    # Ensure we use the correct columns
    df = df[['text', 'label']]
    
    # Drop rows with missing text or labels
    df.dropna(inplace=True)
    
    print(f"Dataset Loaded. Found {len(df)} rows.")
    print(f"Detected Labels: {df['label'].unique()}")
    
    print("Preprocessing data... this may take a minute for 20,000 rows.")
    # Apply your custom preprocessing from utils.py
    df['clean_text'] = df['text'].apply(preprocess_text)
    
except FileNotFoundError:
    print("Error: 'fake_news_dataset.csv' not found.")
    exit()
except KeyError:
    print("Error: Your CSV must have 'text' and 'label' columns.")
    exit()

# -----------------------------
# 3. Feature Extraction (TF-IDF)
# -----------------------------
# Using 5000 features to balance accuracy and performance
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# -----------------------------
# 4. Training
# -----------------------------
# Stratify ensures the split has a balanced ratio of real/fake news
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("\n--- Model Training Complete ---")
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Prediction Function
# -----------------------------
def predict_news(news_text):
    clean_news = preprocess_text(news_text)
    vectorized_news = vectorizer.transform([clean_news])
    prediction = model.predict(vectorized_news)[0]

    # Convert prediction to string and check against known 'real' labels
    pred_str = str(prediction).strip().lower()
    
    if pred_str in ['real', '1', 'true', '1.0']:
        return "🟢 Real News"
    else:
        return "🔴 Fake News"



# -----------------------------
# 7. EXPORT FOR HYBRID SYSTEM
# -----------------------------
# Saving the files so they can be used by app_advanced.py
joblib.dump(model, 'lr_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\n" + "="*40)
print("SUCCESS: 'lr_model.pkl' and 'tfidf_vectorizer.joblib' are ready!")
print("You can now run your Streamlit app.")
print("="*40)


















# # Fake News Detection System using Machine Learning & NLP
# import pandas as pd
# import numpy as np
# import string
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from utils import preprocess_text

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# # -----------------------------
# # 1. Setup & Resources
# # -----------------------------
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# # -----------------------------
# # 2. Text Preprocessing Function
# # -----------------------------
# # def preprocess_text(text):
# #     # Handle non-string inputs and lowercase
# #     text = str(text).lower()
# #     # Remove punctuation
# #     text = text.translate(str.maketrans('', '', string.punctuation))
# #     # Tokenize and remove stopwords/apply stemming
# #     words = text.split()
# #     words = [stemmer.stem(word) for word in words if word not in stop_words]
# #     return " ".join(words)

# # -----------------------------
# # 3. Load & Clean Dataset
# # -----------------------------
# try:
#     # Ensure your file is named exactly this or change the string below
#     df = pd.read_csv("fake_news_dataset.csv")
    
#     # Filter columns and drop rows with missing values
#     # Adjust 'text' and 'label' if your CSV headers are different
#     df = df[['text', 'label']]
#     df.dropna(inplace=True)
    
#     print(f"Dataset Loaded. Found {len(df)} rows.")
#     print(f"Detected Labels: {df['label'].unique()}")
    
#     print("Preprocessing data... please wait.")
#     df['clean_text'] = df['text'].apply(preprocess_text)
    
# except FileNotFoundError:
#     print("Error: 'fack_new.csv' not found. Please place the CSV in this folder.")
#     exit()
# except KeyError:
#     print("Error: Your CSV must have 'text' and 'label' columns.")
#     exit()

# # -----------------------------
# # 4. Feature Extraction (TF-IDF)
# # -----------------------------

# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df['clean_text'])
# y = df['label']

# # -----------------------------
# # 5. Training
# # -----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # -----------------------------
# # 6. Evaluation (Verification)
# # -----------------------------
# y_pred = model.predict(X_test)
# print("\n--- Model Training Complete ---")
# print(f"Validation Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# # -----------------------------
# # 7. Prediction Function
# # -----------------------------
# def predict_news(news_text):
#     # Process the user input same way as training data
#     clean_news = preprocess_text(news_text)
#     vectorized_news = vectorizer.transform([clean_news])
#     prediction = model.predict(vectorized_news)[0]

#     # Convert prediction to string for easier comparison
#     pred_str = str(prediction).strip().lower()
    
#     # Logic: Most datasets use 1/True/Real for positive class
#     if pred_str in ['1', 'real', 'true', '1.0']:
#         return "🟢 Real News"
#     else:
#         return "🔴 Fake News"

# # -----------------------------
# # 8. Interactive Loop
# # -----------------------------
# print("\n" + "="*40)
# print("SYSTEM READY: ENTER TEXT TO TEST")
# print("Type 'exit' or 'quit' to stop.")
# print("="*40)

# while True:
#     user_input = input("\nEnter news article text:\n").strip()
    
#     # Exit condition
#     if user_input.lower() in ['quit', 'exit', 'q']:
#         print("Exiting... Goodbye!")
#         break
        
#     # Validation for empty input
#     if len(user_input) < 5:
#         print("Error: Please enter a longer text for analysis.")
#         continue
    
#     # Get and print result
#     result = predict_news(user_input)
#     print(f"\nResult: {result}")
#     print("-" * 30)

# # -----------------------------
# # 9. EXPORT FOR HYBRID SYSTEM
# # -----------------------------
# import joblib

# # Saving the Logistic Regression model
# joblib.dump(model, 'lr_model.pkl')

# # Saving the TF-IDF Vectorizer (important: it contains your vocabulary!)
# joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# print("\n" + "="*40)
# print("SUCCESS: 'lr_model.pkl' and 'tfidf_vectorizer.joblib' are ready!")
# print("You can now run your Streamlit app.")
# print("="*40)