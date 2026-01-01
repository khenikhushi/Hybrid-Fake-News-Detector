import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure these are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Lowercase and clean
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Tokenize and Lemmatize
    words = text.split()
    # Replace stemmer.stem with lemmatizer.lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)


