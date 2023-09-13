import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_measure(vector1, vector2):
    if len(vector1.shape) == 1:
        vector1 = vector1.reshape(1, -1)
    if len(vector2.shape) == 1:
        vector2 = vector2.reshape(1, -1)
    similarity = cosine_similarity(vector1, vector2)[0, 0]

    return similarity

vector1 = np.array([0.2, 0.4, 0.8])
vector2 = np.array([0.1, 0.7, 0.3])

similarity_score = cosine_similarity_measure(vector1, vector2)

print(f"Cosine Similarity Score: {similarity_score:.2f}")
