from keybert import KeyBERT
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Load models once globally
kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))

def extract_top_keywords(texts, top_n=1):
    keywords = []
    for text in texts:
        try:
            top_kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                use_maxsum=True,
                nr_candidates=20,
                top_n=top_n
            )
            keywords.extend([kw[0] for kw in top_kws])
        except:
            continue
    return keywords

def group_keywords(keywords, num_clusters=5):
    if not keywords:
        return {"No keywords found": []}

    # Vectorize keywords using sentence embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(keywords)

    num_clusters = min(len(keywords), num_clusters) if len(keywords) > 1 else 1

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    df = pd.DataFrame({'keyword': keywords, 'cluster': labels})
    clusters = {}

    for label in df['cluster'].unique():
        theme = f"Cluster {label + 1}"
        clusters[theme] = df[df['cluster'] == label]['keyword'].tolist()

    return clusters
