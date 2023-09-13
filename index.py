from nltk.tokenize import word_tokenize
import nltk
import string
# nltk.download('punkt')
# nltk.download('stopwords')
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
# import pandas as pd

def clean_text(text):
    cleaned_text = ''.join([chars for chars in text if chars not in string.punctuation and not chars.isdigit()])
    return cleaned_text

# Path of the directory/Corpus 
nasa_corpus_dir = Path("C:/Users/Pavan Kumar/Desktop/TCS-IIITH/IRE/Assignment-1/nasa")
text_files = [file for file in nasa_corpus_dir.glob("*.txt")]
tokenized_texts = []
stop_words = set(stopwords.words('english'))

# Process each text file
for file in text_files[:50]:  # example 15 files, 
    with open(file, "r") as f:
        text = f.read()
        cleaned_text = clean_text(text)
        tokens = word_tokenize(cleaned_text)
        tokenized_texts.extend(tokens)

# tokenized texts
# for i, tokens in enumerate(tokenized_texts):
#     print(tokens)

# Perform stemming using Porter algorithm
stemmer = PorterStemmer()
for word in tokenized_texts:
    stemmed_tokens = stemmer.stem(word)
    # print(stemmed_tokens)
# # Calculate word frequencies
fdist = FreqDist(stemmed_tokens)
# print(fdist)
# # # ----
word_freq = Counter(tokenized_texts)
# print(word_freq)
# Print the 20 most frequent words
most_common_words = {}
for word, freq in word_freq.most_common(50):
    # print(f"{word}: {freq}")
    most_common_words[word] = freq
# print(most_common_words)
# ---------
# Get the 50 most frequent words
# most_common_words = fdist.most_common(50)
# print(most_common_words)
# Plot a word cloud for the 50 most frequent words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(most_common_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Tag Cloud for 50 Most Frequent Words")
plt.show()

documents = [text]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

stemmed_filtered_tokens = [stemmer.stem(token) for token in filtered_tokens]
# print(stemmed_filtered_tokens)
# ---------
filtered_freq_dist = FreqDist(stemmed_filtered_tokens)
filtered_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(filtered_freq_dist)

plt.figure(figsize=(10, 5))
plt.imshow(filtered_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# ---------
# ---------
term_frequency = {}
total_terms = len(filtered_tokens)

for term in filtered_tokens:
    if term in term_frequency:
        term_frequency[term] += 1
    else:
        term_frequency[term] = 1

normalized_tf = {term: freq / total_terms for term, freq in term_frequency.items()}


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(filtered_tokens)])  

tfidf_values = tfidf_matrix.toarray()[0]
document_vector = {}
for term in filtered_tokens:
    document_vector[term] = 1


article_tokens = stemmed_tokens  
article_fdist = FreqDist(article_tokens)
top_20_words = article_fdist.most_common(20)

plt.figure(figsize=(12, 6))
article_fdist.plot(20, cumulative=True)
plt.title("Frequency Distribution of the 20 Most Occurring Words in the First NASA Article")
plt.xlabel("Words")
plt.show()
plt.ylabel("Frequency")
plt.show()
num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_values)

cluster_labels = kmeans.labels_

for i, label in enumerate(cluster_labels):
    print(f"Document {i + 1} is assigned to Cluster {label + 1}")

clustered_documents = [[] for _ in range(num_clusters)]

for i, label in enumerate(cluster_labels):
    clustered_documents[label].append(i)

for i, documents_in_cluster in enumerate(clustered_documents):
    print(f"Cluster {i + 1} contains documents: {documents_in_cluster}")

