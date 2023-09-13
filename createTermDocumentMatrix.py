import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import binarize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

# stopwords and punkt tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

#Porter Stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

#text preprocessing
def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    words = [word.lower() for word in words if word.lower() not in stop_words]
    words = [stemmer.stem(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text

def create_term_document_matrix(corpus_dir):
    documents = []
    filenames = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(corpus_dir, filename), 'r') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text)
                documents.append(preprocessed_text)
                filenames.append(filename)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    binary_matrix = binarize(tfidf_matrix)
    tfidf_transformer = TfidfTransformer()
    tfidf_scores = tfidf_transformer.fit_transform(tfidf_matrix)
    return binary_matrix, filenames, vectorizer, tfidf_scores