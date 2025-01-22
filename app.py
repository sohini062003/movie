from flask import Flask, request, render_template, redirect, url_for
import requests
import pickle
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import SnowballStemmer

app = Flask(__name__)

movies = pickle.load(open('movie_list.pkl', 'rb'))
# similarity = pickle.load(open('similarity.pkl', 'rb'))
similarity=joblib.load(open('similarity2.joblib', 'rb'))
bnb= joblib.load(open('bnb.pkl', 'rb'))
cv= joblib.load(open('cv.pkl', 'rb'))




def fetch_reviews(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()
    reviews = [review['content'] for review in response.get('results', [])]
    return reviews

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def rem_stopwords(text):
    
     stop_words = set(stopwords.words('english'))
     words = word_tokenize(text)
     return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

def predict_sentiment_from_api(movie_id):
    reviews = fetch_reviews(movie_id)
    cleaned_reviews = []
    for review in reviews:
        review = clean(review)
        review = is_special(review)
        review = review.lower()
        review = rem_stopwords(review)
        review = stem_txt(review)
        cleaned_reviews.append(review)

    # Transform reviews into feature vectors using the vectorizer
    transformed_reviews = cv.transform(cleaned_reviews).toarray()

    # Predict sentiment using the trained model
    predictions = bnb.predict(transformed_reviews)

    # Return the sentiment (positive or negative) for each review
    return [
        {"review": original, "sentiment": 'positive' if sentiment == 1 else 'negative'}
        for original, sentiment in zip(reviews, predictions)
    ]


# Function to fetch genre list from TMDb API
def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()
    genres = {genre['id']: genre['name'] for genre in response.get('genres', [])}
    return genres

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    return f"https://image.tmdb.org/t/p/w500/{data.get('poster_path')}" if 'poster_path' in data else None

# Function to fetch detailed movie information
# Function to fetch detailed movie information
# Function to fetch detailed movie information
def fetch_movie_details(movie_id):
    # Endpoints
    movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
    certification_url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates?api_key={API_KEY}"
    providers_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"

    # Fetch data
    movie_data = requests.get(movie_url).json()
    credits_data = requests.get(credits_url).json()
    videos_data = requests.get(videos_url).json()
    certification_data = requests.get(certification_url).json()
    providers_data = requests.get(providers_url).json()

    streaming_providers = providers_data.get("results", {}).get("US", {})

    # Extract main cast (top 5) and director
    # cast = [member['name'] for member in credits_data.get("cast", [])[:5]] or ["Not Available"]
    cast = [
        {
            "name": member['name'],
            "character": member.get('character', 'Unknown'),
            "profile": f"https://image.tmdb.org/t/p/w200{member['profile_path']}" if member['profile_path'] else None
        }
        for member in credits_data.get("cast", [])[:5]  # Get top 5 cast members
    ] or ["Not Available"]
    director = next((member['name'] for member in credits_data.get("crew", []) if member['job'] == 'Director'), "Not Available")

    # Extract age certification
    certifications = certification_data.get('results', [])
    certification = "Not Available"
    for cert in certifications:
        if cert.get('iso_3166_1') == 'US':  # Adjust region as needed
            certification = cert.get('release_dates', [{}])[0].get('certification', "Not Available")
            break

    # Extract trailer
    trailer = None
    for video in videos_data.get("results", []):
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            trailer = f"https://www.youtube.com/watch?v={video['key']}"
            break
    trailer = trailer or "Not Available"

    # Compile details
    return {
        "title": movie_data.get("title", "Not Available"),
        "overview": movie_data.get("overview", "Not Available"),
        "poster": f"https://image.tmdb.org/t/p/w500/{movie_data.get('poster_path')}" if movie_data.get("poster_path") else None,
        "release_date": movie_data.get("release_date", "Not Available"),
        "rating": movie_data.get("vote_average", "Not Available"),
        "runtime": movie_data.get("runtime", "Not Available"),
        "budget": movie_data.get("budget", "Not Available"),
        "revenue": movie_data.get("revenue", "Not Available"),
        "genres": [genre['name'] for genre in movie_data.get("genres", [])] or ["Not Available"],
        "production_companies": [company['name'] for company in movie_data.get("production_companies", [])] or ["Not Available"],
        "spoken_languages": [language['name'] for language in movie_data.get("spoken_languages", [])] or ["Not Available"],
        "tagline": movie_data.get("tagline", "Not Available"),
        "keywords": [keyword['name'] for keyword in movie_data.get("keywords", {}).get("keywords", [])] or ["Not Available"],
        "popularity": movie_data.get("popularity", "Not Available"),
        "homepage": movie_data.get("homepage", "Not Available"),
        "cast": cast,
        "crew": [
            {"name": crew['name'], "job": crew['job']}
            for crew in credits_data.get("crew", [])[:5]
        ] or ["Not Available"],
        "director": director ,
        "certification": certification,
        "trailer": trailer ,
        "streaming": [
            provider["provider_name"]
            for provider in streaming_providers.get("flatrate", [])
        ] or ["Not Available"],
        "rent": [
            provider["provider_name"]
            for provider in streaming_providers.get("rent", [])
        ] or ["Not Available"],
        "buy": [
            provider["provider_name"]
            for provider in streaming_providers.get("buy", [])
        ] or ["Not Available"],
        "backdrop": f"https://image.tmdb.org/t/p/w1280/{movie_data.get('backdrop_path')}" if movie_data.get("backdrop_path") else None,
    }



# Function to get recommended movies based on movie title
def get_recommendations(movie):
    try:
        # Get the index of the movie
        idx = movies[movies['title'] == movie].index
        if len(idx) == 0:  # Movie not found in dataset
            return [], [], []
        idx = idx[0]

        # Get similarity scores and sort them
        sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:6]

        # Extract indices of the top 5 movies
        movie_indices = [i[0] for i in sim_scores]

        # Fetch titles, posters, and IDs
        movie_titles = movies['title'].iloc[movie_indices].tolist()
        movie_posters = [fetch_poster(movies['movie_id'].iloc[i]) for i in movie_indices]
        movie_ids = movies['movie_id'].iloc[movie_indices].tolist()
        return movie_titles, movie_posters, movie_ids
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return [], [], []


# Function to get movies by genre
# Function to get movies by genre
def get_movies_by_genre(genre_id):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre_id}&language=en-US"
    response = requests.get(url).json()
    movies = response.get('results', [])
    
    movie_titles = [movie['title'] for movie in movies]
    movie_posters = [f"https://image.tmdb.org/t/p/w500/{movie.get('poster_path')}" if 'poster_path' in movie else None for movie in movies]
    movie_ids = [movie['id'] for movie in movies]  # Include movie IDs
    
    return movie_titles, movie_posters, movie_ids


# Main page route
# Main page route
@app.route('/', methods=['GET', 'POST'])
def index():
    movie_list = movies['title'].tolist()
    genres = fetch_genres()  # Fetch genre list for dropdown
    recommended_movie_titles = []
    recommended_movie_posters = []
    recommended_movie_ids = []

    if request.method == 'POST':
        if 'selected_movie' in request.form:
            movie_title = request.form['selected_movie']
            recommended_movie_titles, recommended_movie_posters, recommended_movie_ids = get_recommendations(movie_title)
        elif 'selected_genre' in request.form:
            genre_name = request.form['selected_genre']
            genre_id = next((key for key, value in genres.items() if value.lower() == genre_name.lower()), None)
            if genre_id:
                recommended_movie_titles, recommended_movie_posters, recommended_movie_ids = get_movies_by_genre(genre_id)

    return render_template('index3.html', movie_list=movie_list, genres=genres,
                           recommended_movie_titles=recommended_movie_titles,
                           recommended_movie_posters=recommended_movie_posters,
                           recommended_movie_ids=recommended_movie_ids)

# Route to display sentiment analysis for a movie's reviews
@app.route('/movie/<int:movie_id>/reviews')
def movie_reviews(movie_id):
    sentiments = predict_sentiment_from_api(movie_id)
    movie_details = fetch_movie_details(movie_id)
    return render_template('reviews.html', movie=movie_details, sentiments=sentiments, movie_id=movie_id)


# Movie details page route
@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    details = fetch_movie_details(movie_id)
    return render_template('movie_details.html', details=details, movie_id=movie_id)

@app.route('/about')
def about():
    return render_template('about2.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
