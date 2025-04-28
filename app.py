import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os
# Load all saved models and data
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

song_vectorizer, text_vectors, numeric_vectors, data, tsne_data, kmeans_model, scaler = load_models()

# Set up Streamlit layout
st.set_page_config(page_title="Song Recommendation App", page_icon="🎵")
st.title("Song Recommendation App")
st.text("Enter a song name to get similar song recommendations.")

# User input for song name
song_name = st.text_input("Song Name:")

# Button to trigger recommendation
if st.button("Get Recommendations"):
    if song_name:
        try:
            if song_name in data['name'].values:
                recommendations = get_similarities_optimized(song_name, data, text_vectors, numeric_vectors)
                recommendations = recommendations[recommendations['name'] != song_name]  # Exclude input song
                top_recommendations = recommendations.nlargest(5, 'similarity')  # Get top 5 recommendations
                
                st.dataframe(top_recommendations[['name', 'similarity', 'popularity']])
            else:
                st.error("Song not found. Here are 5 random popular songs:")
                random_songs = data.sample(5)
                st.dataframe(random_songs[['name', 'popularity']])
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a song name.")

# Optional t-SNE Visualization
with st.sidebar.expander("Explore Dataset (t-SNE)", expanded=False):
    if tsne_data is not None:
        st.write("t-SNE Visualization of Songs")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=kmeans_model.labels_ if kmeans_model else None, palette='viridis', alpha=0.7)
        plt.title("t-SNE Visualization of Songs")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        st.pyplot(plt)
    else:
        st.write("t-SNE data not available.")

# Function to get similar songs
def get_similarities_optimized(song_name, data, text_vectors, numeric_vectors):
    # Find the index of the song
    song_index = data[data['name'] == song_name].index[0]
    
    # Calculate cosine similarity
    similarities = cosine_similarity(text_vectors[song_index].reshape(1, -1), text_vectors).flatten()
    
    # Create a DataFrame for recommendations
    recommendations = pd.DataFrame({
        'name': data['name'],
        'similarity': similarities,
        'popularity': data['popularity']
    })
    
    return recommendations

# Footer
st.markdown("---")
st.markdown("Built by Commander")
