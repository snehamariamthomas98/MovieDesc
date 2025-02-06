import pytest
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import isodate

# Sample DataFrame for Testing
data = {
    "duration": ["PT2M30S", "PT5M", "PT1H", np.nan, "PT15M"],
    "page_dir": ["page1", "page2", "page1", "page3", "page2"],
    "name": ["Movie one", "Another movie", "Cool Movie", "Film name", "Some Movie"]
}
df = pd.DataFrame(data)

# Function to test duration conversion
def test_duration_to_seconds():
    def duration_to_seconds(duration):
        if pd.isnull(duration):
            return np.nan
        try:
            parsed_duration = isodate.parse_duration(duration)
            return parsed_duration.total_seconds()
        except Exception:
            return np.nan
    
    df["duration_seconds"] = df["duration"].apply(duration_to_seconds)
    assert df["duration_seconds"].isnull().sum() == 1  # One NaN value expected
    assert df["duration_seconds"].iloc[0] == 150  # PT2M30S = 150s
    assert df["duration_seconds"].iloc[2] == 3600  # PT1H = 3600s

def test_text_preprocessing():
    from nltk.stem import WordNetLemmatizer
    import re
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)
    
    df["cleaned_name"] = df["name"].astype(str).apply(preprocess)
    assert "movie" in df["cleaned_name"].iloc[0]
    assert "another" not in df["cleaned_name"].iloc[1]  # Stop word removed

def test_word_frequency():
    word_freq = Counter(" ".join(df["cleaned_name"]).split())
    assert len(word_freq) > 0  # Ensure some words exist
    assert "movie" in word_freq  # "movie" should be a common term

def test_tfidf_vectorization():
    vectorizer = TfidfVectorizer(max_features=5)
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_name"])
    assert tfidf_matrix.shape[1] == 5  # Max 5 features
    assert tfidf_matrix.shape[0] == df.shape[0]  # Same number of rows

def test_kmeans_clustering():
    vectorizer = TfidfVectorizer(max_features=5)
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_name"])
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    assert len(set(clusters)) == num_clusters  # Ensure correct number of clusters

def test_cosine_similarity():
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["cleaned_name"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    assert similarity_matrix.shape == (df.shape[0], df.shape[0])  # Square matrix
    assert np.all(similarity_matrix.diagonal() == 1)  # Self-similarity should be 1

def test_csv_output():
    df.to_csv("test_output.csv", index=False)
    loaded_df = pd.read_csv("test_output.csv")
    assert loaded_df.shape == df.shape  # Ensure CSV shape is consistent
