import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_query_vector(query, vectorizer, tfidf_matrix, filenames, top_n=5):
# query into a vector representation.
    def preprocess_query(query):
        query_words = query.split()  # Assuming the query is a space-separated string.
        query_words = [word.lower() for word in query_words if word.lower() not in stop_words]
        query_words = [stemmer.stem(word) for word in query_words]
        return ' '.join(query_words)
    query = preprocess_query(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = similarities.argsort(axis=1)[:, -top_n:][:, ::-1]

    for i, idx_row in enumerate(top_indices):
        for j, idx in enumerate(idx_row):
            similarity = similarities[i][idx]
            filename = filenames[idx]
            print(f"{similarity},{filename}")
