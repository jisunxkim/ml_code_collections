# content based recommendation
# Content-Based Recommendation Engine that computes similarity between movies based on movie genres. It will suggest movies that are most similar to a particular movie based on its genre.

import pandas as pd
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import pairwise_distances

import nltk
import string

pd.options.display.max_columns = None
# 1. load data and clean
# movies.dat
# MovieID::Title::Genres
movies = pd.read_csv(

    './datasets/movie_lens_1m/movies.dat',
    sep='::', encoding='ISO-8859-1',
    header=None,
    names=['movie_id', 'title', 'genres'])

movies.head() # 0  1  Toy Story (1995)   Animation|Children's|Comedy

# 2. EDA
# List top 5 genres
movies['genres_list'] = (
    movies['genres'].apply(lambda x: x.lower().split('|'))
)
popularity_genres = dict()
genres = movies['genres_list'].to_list()
for genres in movies['genres_list'].to_list():
    for genre in genres: 
        popularity_genres[genre] = 1 + popularity_genres.get(genre, 0)

popularity_genres = list(popularity_genres.items())
popularity_genres = sorted(
    popularity_genres, 
    key=lambda x: x[1],
    reverse=True
)     
print(popularity_genres[:5]) # [('drama', 1603), ('comedy', 1200), ('action', 503), ('thriller', 492), ('romance', 471)]

# 3. Item profile - tfidf feature vectors of genres
# tfidf_matrix =>(docs, terms) matrix
tfidf = TfidfVectorizer(
    analyzer='word', ngram_range=(1,2), min_df=0.0,
    stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres']) 

print(tfidf.get_feature_names_out()) # 1 and 2 grams words
print(tfidf.get_feature_names_out().shape) # (127,)
print(tfidf_matrix.toarray().shape) # (3883, 127)

# tfidf with custom text cleaning
def tokenize_and_lemmatize(text):
    # special case: the data include '|' to seperate words
    text = text.replace('|', ' ')
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text =  text.translate(translator)
    # tokenize
    tokens = nltk.tokenize.word_tokenize(text)
    # remove stopwords and lemmatizer
    stop_words = set(nltk.corpus.stopwords.words('english')) 
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens
                         if token not in stop_words]
    return lemmatized_tokens

tfidf_lemmatizer = TfidfVectorizer(
    tokenizer=tokenize_and_lemmatize,
    analyzer='word', ngram_range=(1,1), min_df=0.0,
    )
tfidf_lemmatizer_matrix = tfidf_lemmatizer.fit_transform(movies['genres']) # return 3883x302 sparse matrix of type '<class 'numpy.float64'>'
tfidf_lemmatizer.get_feature_names_out()
# convert the sparse matrix to dense matrix
tfidf_lemmatizer_matrix = tfidf_lemmatizer_matrix.toarray() # (3883, 302) 
tfidf_lemmatizer.get_feature_names_out()

"my sci's story|drama good,movie".replace(('|', ','), ' ')
 
# 4. calulcate similarity between two movies 
# using cosine similarity
# Since I have used the TF-IDF Vectorizer, calculating the Dot Product will directly give me the Cosine Similarity Score. Therefore, I will use sklearn’s linear_kernel instead of cosine_similarities since it is much faster.
# Mathematically, the linear kernel between two vectors 
# u and v is defined as the dot product of the two vectors.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape) # (3883, 3883)

cosine_sim2 = 1- pairwise_distances(tfidf_matrix, metric='cosine')
print(cosine_sim2.shape) # (3883, 3883)

 
# 5. recommender based on genres
# The next step is to write a function that returns 
# the 20 most similar movies based on the cosine similarity score.

titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])
def genre_recommendation(title):
    if title.lower() not in indices.index.str.lower():
        return
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21] # 0 is always itself. Therefore start from 1!
    movie_indices = [score[0] for score in sim_scores]
    return titles.iloc[movie_indices]

genre_recommendation('Good Will Hunting (1997)')


from sklearn.model_selection import train_test_split

temp = train_test_split(movies, test_size=0.2)