import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_query(query):
    query_words = word_tokenize(query)
    query_words = [word for word in query_words if word.isalpha()]
    query_words = [word.lower() for word in query_words if word.lower() not in stop_words]
    query_words = [stemmer.stem(word) for word in query_words]
    boolean_query = set(query_words)
    return boolean_query
