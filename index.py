import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set the random seed
np.random.seed(5)

# Download NLTK resources
nltk.download('punkt')

# Read the movies data from CSV
movies_df = pd.read_csv('Mov_project data.csv')

# Define tokenization and stemming function
def tokenize_and_stem(text):
    stemmer = SnowballStemmer('english')
    tokens = [words for sent in nltk.sent_tokenize(text) for words in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    stem = [stemmer.stem(words) for words in filtered_tokens]
    return stem

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words=None,
                                   use_idf=True, tokenizer=tokenize_and_stem,
                                   ngram_range=(1, 3))

# Apply TF-IDF transformation to movie plots
tfidf_matrix = tfidf_vectorizer.fit_transform(x for x in movies_df['Plot'])

# Compute cosine similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Function to recommend movies
def recommend_movies(title):
    global movies_df,similarity_distance

    if title in movies_df['Title'].values:
        # Movie is present in the database
        ind = movies_df.loc[movies_df['Title'] == title].index[0]
        mov_indices = np.argsort(similarity_distance[ind])[1:6]
        recommended_movies = movies_df.loc[mov_indices]['Title']
    else:
        # Movie is not present in the database, scrape description, release year, and genre and add them to the database
        description, release_year, genre = search_movie_details(title)
        new_movie = pd.DataFrame([[release_year, title, genre, description]],
                                 columns=movies_df.columns)
        updated_movies_df = pd.concat([movies_df, new_movie], ignore_index=True)
        updated_tfidf_matrix = tfidf_vectorizer.fit_transform(x for x in updated_movies_df['Plot'])
        similarity_distance = 1 - cosine_similarity(updated_tfidf_matrix)
        ind = updated_movies_df.loc[updated_movies_df['Title'] == title].index[0]
        mov_indices = np.argsort(similarity_distance[ind])[1:6]
        recommended_movies = updated_movies_df.loc[mov_indices]['Title']
        movies_df = updated_movies_df

    return recommended_movies

# Function to search for movie details (description, release year, genre) from Google search results
def search_movie_details(movie_title):
    query = f"{movie_title} movie"
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Scrape movie description
    search_result = soup.find('div', class_='BNeawe')
    description = search_result.text if search_result else "Description not found"

    # Scrape movie release year
    release_year = ""
    release_year_element = soup.find('span', class_='sd')
    if release_year_element:
        release_year = release_year_element.text.strip()

    # Scrape movie genre
    genre = ""
    genre_element = soup.find('span', class_='w8qArf')
    if genre_element:
        genre = genre_element.text.strip()

    return description, release_year, genre

# Prompt for movie title input
movie_title = input("Enter a movie title: ")
recommended_movies = recommend_movies(movie_title)

if len(recommended_movies) > 0:
    print("Recommended Movies:")
    for movie in recommended_movies:
        print(movie)

# Save the updated database to CSV
movies_df.to_csv('Mov_project data.csv', index=False)
