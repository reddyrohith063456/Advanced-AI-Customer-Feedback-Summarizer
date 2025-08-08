import streamlit as st
import pandas as pd
import openai
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import chardet

# Load OpenAI API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Sentiment Analysis ---
def detect_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --- Keyword Clustering ---
def extract_keywords_and_cluster(df, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['feedback'].astype(str))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    df['cluster'] = kmeans.labels_

    keywords_per_cluster = []
    for i in range(num_clusters):
        cluster_indices = (kmeans.labels_ == i)
        cluster_data = X[cluster_indices]
        avg_tfidf = cluster_data.mean(axis=0).A1
        top_indices = avg_tfidf.argsort()[::-1][:5]
        top_keywords = [vectorizer.get_feature_names_out()[j] for j in top_indices]
        keywords_per_cluster.append((i, top_keywords))
    return df, keywords_per_cluster

# --- Streamlit UI ---
st.set_page_config(page_title="GenAI Feedback Analyzer")
st.title("GenAI Customer Feedback Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Detect encoding
    raw_data = uploaded_file.read()
    encoding = chardet.detect(raw_data)['encoding']
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Could not read file even with fallback. Error: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    # Check required columns
    required_columns = {"feedback", "rating", "product"}
    if not required_columns.issubset(df.columns):
        st.error("Your file must contain 'feedback', 'rating', and 'product' columns.")
        st.stop()

    # Sentiment
    with st.spinner("Analyzing sentiment..."):
        df['sentiment'] = df['feedback'].apply(detect_sentiment)

    st.subheader("Sample Preview")
    st.dataframe(df.head())

    # Sentiment Chart
    st.subheader("Sentiment Analysis")
    sentiment_counts = df['sentiment'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color='green')
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Ratings Chart
    st.subheader("Overall Rating Distribution")
    rating_counts = df['rating'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(rating_counts.index, rating_counts.values, color='skyblue')
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # Product-wise Chart
    st.subheader("Product-wise Rating Analysis")
    unique_products = df['product'].unique()
    selected_product = st.selectbox("Select product:", options=np.append(["All"], unique_products))

    if selected_product == "All":
        for product in unique_products:
            product_data = df[df['product'] == product]
            st.markdown(f"### Product: `{product}`")
            product_ratings = product_data['rating'].value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(product_ratings.index, product_ratings.values, color='orange')
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            ax.set_title(f"Rating Breakdown for {product}")
            st.pyplot(fig)
    else:
        product_data = df[df['product'] == selected_product]
        st.markdown(f"### Product: `{selected_product}`")
        product_ratings = product_data['rating'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(product_ratings.index, product_ratings.values, color='orange')
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title(f"Rating Breakdown for {selected_product}")
        st.pyplot(fig)

    # Keyword filter
    st.subheader("Filter Feedback by Keyword")
    keyword = st.text_input("Enter keyword to filter:")
    if keyword:
        filtered = df[df['feedback'].str.contains(keyword, case=False)]
        st.write(f"Showing {len(filtered)} feedbacks with keyword '{keyword}':")
        st.dataframe(filtered[['customer_id', 'product', 'rating', 'feedback']])

    # Thematic Clustering
    st.subheader("Thematic Clustering")
    df, keyword_clusters = extract_keywords_and_cluster(df)
    for cluster_id, keywords in keyword_clusters:
        st.markdown(f"**Cluster {cluster_id + 1}:** `{', '.join(keywords)}`")
        cluster_feedback = df[df['cluster'] == cluster_id]['feedback'].head(3).tolist()
        st.json(cluster_feedback)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full CSV with Sentiment & Clusters", csv, "enhanced_feedback.csv", "text/csv")

else:
    st.info("Upload a CSV file with 'feedback', 'rating', and 'product' columns.")
