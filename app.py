# -------------------------------
# Import Libraries
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Models and Data
# -------------------------------
@st.cache_resource
def load_models():
    song_vectorizer = joblib.load('song_vectorizer.pkl')
    text_vectors = joblib.load('text_vectors.pkl')
    numeric_vectors = joblib.load('numeric_vectors.pkl')
    data = joblib.load('data.pkl')
    tsne_data = joblib.load('tsne_data.pkl') if 'tsne_data.pkl' in os.listdir() else None
    kmeans_model = joblib.load('kmeans_model.pkl') if 'kmeans_model.pkl' in os.listdir() else None
    scaler = joblib.load('scaler.pkl') if 'scaler.pkl' in os.listdir() else None
    return song_vectorizer, text_vectors, numeric_vectors, data, tsne_data, kmeans_model, scaler

# Load all artifacts
song_vectorizer, text_vectors, numeric_vectors, data, tsne_data, kmeans_model, scaler = load_models()

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="🎵 MelodyMind - Song Recommender", page_icon="🎶")
st.title("🎵 MelodyMind - Intelligent Song Recommendation App")
st.markdown("Enter a song name and get intelligent music recommendations based on audio features!")

# -------------------------------
# User Input Section
# -------------------------------
song_name = st.text_input("Enter a song name:", "")

# -------------------------------
# Recommendation Logic
# -------------------------------
def get_similarities_optimized(song_name, data, text_vectors, numeric_vectors):
    # Find song index
    song_indices = data.index[data['name'] == song_name].tolist()
    if not song_indices:
        raise ValueError(f"Song '{song_name}' not found in the dataset.")
    song_index = song_indices[0]

    # Calculate similarities
    text_array1 = text_vectors[song_index]
    num_array1 = numeric_vectors[song_index]

    text_sim = cosine_similarity([text_array1], text_vectors).flatten()
    num_sim = cosine_similarity([num_array1], numeric_vectors).flatten()

    total_similarity = text_sim + num_sim

    # Create DataFrame
    recommendations = pd.DataFrame({
        'name': data['name'],
        'similarity': total_similarity,
        'popularity': data['popularity']
    })

    return recommendations

# -------------------------------
# Recommendation Output
# -------------------------------
if st.button("Get Recommendations"):
    if song_name:
        try:
            recommendations = get_similarities_optimized(song_name, data, text_vectors, numeric_vectors)
            recommendations = recommendations[recommendations['name'] != song_name]  # Exclude input song
            top_recommendations = recommendations.sort_values(by=['similarity', 'popularity'], ascending=[False, False]).head(5)
            st.success(f"Top 5 Recommendations similar to '{song_name}':")
            st.dataframe(top_recommendations[['name', 'similarity', 'popularity']])
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Here are 5 random popular songs you may like:")
            random_songs = data.sample(5)
            st.dataframe(random_songs[['name', 'popularity']])
    else:
        st.warning("Please enter a valid song name!")

# -------------------------------
# Optional Sidebar t-SNE Visualization
# -------------------------------
with st.sidebar.expander("🎨 Explore Dataset (t-SNE Map)", expanded=False):
    if tsne_data is not None:
        st.write("Visualizing songs in 2D space colored by clusters.")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Downsampled version for plotting
        sample_size = min(len(tsne_data), 5000)  # Adjust if needed
        sample_clusters = data.sample(n=sample_size, random_state=42)['cluster'].values
        
        sns.scatterplot(
            x=tsne_data[:, 0],
            y=tsne_data[:, 1],
            hue=sample_clusters,
            palette='viridis',
            alpha=0.7,
            ax=ax
        )
        plt.title("t-SNE Visualization of Songs by Clusters")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
    else:
        st.info("t-SNE data not available.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Built with ❤️ by Commander.")
