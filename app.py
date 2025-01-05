# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
raw_data = pd.read_csv("imdb_top_1000.csv", index_col=False)

# Get the relevant features only
new_data = raw_data[["Series_Title", "Genre", "IMDB_Rating", "Overview", "Star1", "Star2", "Star3", "Star4", "Poster_Link"]]

# Convert genre to list
new_data["Genre"] = new_data["Genre"].apply(lambda x: x.split(", "))

# Do one-hot encoding for genre
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(new_data["Genre"])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# Preprocess IMDB Rating
scaler = MinMaxScaler()
new_data["IMDB_Rating"] = scaler.fit_transform(new_data[["IMDB_Rating"]])

# Preprocess Overview
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(new_data["Overview"])

# Preprocess actors
new_data["Actors"] = new_data[["Star1", "Star2", "Star3", "Star4"]].agg(" ".join, axis=1)
actor_tfidf = TfidfVectorizer(stop_words="english")
actor_matrix = actor_tfidf.fit_transform(new_data["Actors"])

# Combine all features
tfidf_dense = tfidf_matrix.toarray()
actor_dense = actor_matrix.toarray()
combined_features = np.hstack([
    genre_matrix,
    new_data["IMDB_Rating"].values.reshape(-1, 1),
    tfidf_dense,
    actor_dense
])

# Compute cosine similarity
similarity_matrix = cosine_similarity(combined_features, combined_features)

# Recommendation function
def recommend(movie_title, n_recommendation):
    try:
        idx = new_data[new_data["Series_Title"] == movie_title].index[0]
    except IndexError:
        return f"Movie {movie_title} not found in the dataset"
    
    similarity_score = list(enumerate(similarity_matrix[idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    top_movies = [(new_data["Series_Title"][i[0]], i[1]) for i in similarity_score[1:n_recommendation + 1]]
    
    return top_movies

# Streamlit UI
st.title("Movie Recommendation System")

# Dropdown for selecting a movie (sorted alphabetically)
sorted_movies = sorted(new_data["Series_Title"])
selected_movie = st.selectbox("Select a Movie", sorted_movies)

# Slider for number of recommendations
num_recommendations = st.slider("Number of Recommendations", 1, 5, 3)

# Button to show recommendations
if st.button("Get Recommendations"):
    # Display the selected movie's image
    selected_movie_img = new_data[new_data["Series_Title"] == selected_movie]["Poster_Link"].values[0]
    st.image(selected_movie_img, caption=selected_movie, width=200)

    # Get top recommendations
    recommendations = recommend(selected_movie, num_recommendations)
    
    for movie, overview, score in recommendations:
        movie_img = new_data[new_data["Series_Title"] == movie]["Poster_Link"].values[0]
        st.image(movie_img, caption=f"{movie} (Score: {score:.2f})", width=200)
        st.write(f"**Overview**: {overview}")
