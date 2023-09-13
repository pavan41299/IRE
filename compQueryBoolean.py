import numpy as np

def compare_query_boolean(query, binary_matrix, filenames, top_n=5):
    # Convert the query into a boolean representation.
    def preprocess_query(query):
        # Tokenize and preprocess the query (similar to preprocessing documents).
        query_words = query.split()
        query_words = [word.lower() for word in query_words if word.lower() not in stop_words]
        query_words = [stemmer.stem(word) for word in query_words]
        return ' '.join(query_words)

    query = preprocess_query(query)
    query_vector = vectorizer.transform([query])
    query_binary_vector = binarize(query_vector)
    similarities = np.dot(binary_matrix, query_binary_vector.T)
    top_indices = similarities.argsort(axis=0)[-top_n:][::-1]

    for idx in top_indices:
        similarity = similarities[idx][0]
        filename = filenames[idx[0]]
        print(f"{similarity}, {filename}")