import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_query(query):
    query_words = word_tokenize(query)
    query_words = [word for word in query_words if word.isalpha()]
    query_words = [word.lower() for word in query_words if word.lower() not in stop_words]
    query_words = [stemmer.stem(word) for word in query_words]
    preprocessed_query = ' '.join(query_words)
    return preprocessed_query

def create_vector_query(query, vectorizer):
    preprocessed_query = preprocess_query(query)
    query_vector = vectorizer.transform([preprocessed_query])
    return query_vector