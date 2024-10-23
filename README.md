# Movie Recommendation System

## Overview
This project is a movie recommendation system that uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize movie plots and **cosine similarity** to recommend movies similar to a given title. If the movie is not in the dataset, the system automatically scrapes movie details (description, release year, and genre) using Google search results and adds it to the dataset. 

## Features
- **TF-IDF Vectorizer**: Converts movie plots into numerical data.
- **Cosine Similarity**: Recommends 5 movies most similar to the input movie.
- **Web Scraping**: Automatically searches for movie details if the movie is not found in the dataset.
- **Movie Data Update**: Newly found movies are added to the dataset and saved.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: Used for handling and manipulating the movie dataset.
- **NLTK**: Used for tokenization and stemming of the movie plots.
- **BeautifulSoup**: Used for scraping movie details from Google search results.
- **TF-IDF Vectorizer**: Used to convert movie plots to a numerical matrix.
- **Cosine Similarity**: Used to find similar movies based on their plot.

## Prerequisites
To run this project, you'll need to have the following Python libraries installed:
```bash
pip install numpy pandas nltk scikit-learn requests beautifulsoup4 tensorflow
```
Additionally, download NLTK's 'punkt' resource, which is used for tokenizing sentences:
```python
import nltk
nltk.download('punkt')
```

## How to Run
1. **Prepare the dataset**: Ensure that the `Mov_project data.csv` file is present in the same directory as the script.
2. **Run the script**: Execute the script to start the movie recommendation system. You will be prompted to enter a movie title.
3. **Recommendations**: If the movie is in the dataset, the system will display 5 similar movie recommendations. If the movie is not in the dataset, it will scrape Google for movie details, add it to the dataset, and then provide recommendations.
4. **Updated Dataset**: After each session, the updated movie dataset is saved back to `Mov_project data.csv`.

### Example
```bash
Enter a movie title: The Matrix
Recommended Movies:
- Inception
- Terminator 2: Judgment Day
- The Dark Knight
- Blade Runner
- Total Recall
```

## Functions Breakdown
### 1. `tokenize_and_stem(text)`
- Tokenizes text into words and stems them using the **Snowball Stemmer**.

### 2. `recommend_movies(title)`
- Recommends 5 movies based on their cosine similarity to the input movie title. If the movie is not in the dataset, it scrapes the web for details and updates the dataset.

### 3. `search_movie_details(movie_title)`
- Scrapes Google search results to retrieve the movie's description, release year, and genre.

## Future Improvements
- Enhance scraping to be more robust and extract additional details like movie ratings or actors.
- Add a user interface for ease of use.
- Improve the search algorithm to handle alternate titles and misspellings.

---
