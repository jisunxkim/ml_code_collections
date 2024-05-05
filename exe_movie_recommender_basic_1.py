# https://www.geeksforgeeks.org/recommendation-system-in-python/
# Implementation of Recommendation System
import numpy as np
import pandas as pd
import sklearn 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.max_columns = None 

# loading rating data
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
print(ratings.head())

movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
print(movies.head())

# Statistical Analysis of Ratings
print("\n","*"*20)
print("Statistical Analysis of Ratings")
n_ratings = len(ratings)
n_movies = movies['movieId'].nunique()
n_users = ratings['userId'].nunique()
num_ratings_per_user = round(n_ratings / n_users, 2)
num_ratings_per_movie = round(n_ratings / n_movies, 2)
print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique userId's: {n_users}")
print(f"Average number of ratings per user: {num_ratings_per_user}")
print(f"Average ratings per movies: {num_ratings_per_movie}")

# User Rating Frequency
print("\n","*"*20)
print("User Rating Frequency")
user_freq = (
    ratings
    .groupby('userId', as_index=False)
    .agg(n_ratings = ('rating', 'count'))
)
print(user_freq.head())

# Movie Rating Analysis
# Determine which movies in the dataset have the lowest and highest ratings
movie_ratings = (
    ratings
    .groupby('movieId')
    [['rating']]
    .agg(movie_rating = ('rating', 'mean'))
)
lowest_rated = movie_ratings.idxmin()
highest_rated = movie_ratings.idxmax()
print("lowest rated movie: \n", 
      movies.set_index('movieId').loc[lowest_rated])
print("highest rated movie: \n", 
      movies.loc[movies.movieId.isin(highest_rated)])
# statistics of rating counts and average rating of each movie
movie_stats = (
    ratings
    .groupby('movieId', as_index=False)
    [['rating']]
    .agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    )
    .sort_values(by=['rating_count', 'avg_rating'],
                  ascending=[False, False])
    .merge(movies, on='movieId', how='left')
)

print("average rating and rating count: \n", movie_stats.head())


# User-Item Matrix Creation: (M, N) ; M: movies, N: users
def create_matrix(df):
    """
    A sparse matrix X is created using the SciPy function csr_matrix. The user and movie indices that correspond to the rating values in the dataset are used to generate this matrix. The form of it is (M, N), where M denotes the quantity of distinct films and N denotes the quantity of distinct consumers.
    """
    # user_mapper: userId to index of the matrix
    user_mapper = dict(
        [(uid, idx) for idx, uid in 
            enumerate(sorted(df['userId'].unique()))]
        )
    # user_inv_mapper: index to userId
    user_inv_mapper = dict(
        enumerate(sorted(ratings['userId'].unique()))
    )
    # movie_mapper: movieId to index of the matrix
    movie_mapper = dict(
        [(mid, idx) for idx, mid in 
            enumerate(sorted(df['movieId'].unique()))]
        )
    # movie_inv_mapper: index to movieId
    movie_inv_mapper = dict(
        enumerate(sorted(ratings['movieId'].unique()))
    )

    # maxtrix
    # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    N = df['userId'].nunique()
    M = df['movieId'].nunique()
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix(
        (
            df['rating'], 
            (movie_index, user_index)
            ),
        shape=(M, N) # (movie, user)
        ) # (9724, 610)
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Movie Similarity Analysis
print("\n","*"*20)
print("Movie Similarity Analysis")


def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    """
    Find k number of similar movies to movie_id.
    Method: K Nearlest Neighbor (KNN) model based on rating data, X.
        NearestNeighbors implements unsupervised nearest neighbors learning. It acts as a uniform interface to three different nearest neighbors algorithms: BallTree, KDTree, and a brute-force algorithm based on routines in sklearn.metrics.pairwise. The choice of neighbors search algorithm is controlled through the keyword 'algorithm', which must be one of ['auto', 'ball_tree', 'kd_tree', 'brute']. 
    args:
        X: sparse rating matrix of (Movie, User)
        k: number of similar movies, a parameter of KNN   
    results:
        distances: The distances array contains the distances from each point to its nearest neighbors. It is a 2D array where each row corresponds to a point in the dataset, and each column corresponds to one of its nearest neighbors.
        For example, if distances[i, j] contains the distance from point X[i] to its j-th nearest neighbor. The first value is self-distance, so always zero.  
    """

    neighbhor_ids = [] 

    movie_idx = movie_mapper[movie_id]
    movie_vec = X[movie_idx]

    # k include the movie itself, so add one more.
    k += 1

    kNN = NearestNeighbors(
        n_neighbors=k, algorithm='brute', metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    # retureturn_distance = True: return a tuple ([distinace], [index])
    neighbor = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        if not show_distance:
            n = neighbor.item(i)
        else:
            n = neighbor[1].item(i)
        neighbhor_ids.append(movie_inv_mapper[n])
        
    neighbhor_ids.pop(0) # remove the self-movie.
    return neighbhor_ids

movie_titles = dict(zip(movies['movieId'], movies['title']))
movie_id = 3
similar_ids = find_similar_movies(movie_id, X, k=10)
print("Recommending 10 movies based on similarities among the movies")
print(f'Since you watched {movie_titles[movie_id]}: ')
for i in similar_ids:
    print(movie_titles[i])


# Movie Recommendation with respect to Users Preference
print("\n","*"*20)
print("Movie Recommendation with respect to Users Preference")

   
def recommend_movies_for_user(
    user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
    
    # select ratings of the user
    df1 = ratings[ratings['userId'] == user_id]
     
    if df1.empty:
        print(f"User with ID {user_id} does not exist.")
        return
    # select a movie_id that the user ratied the higest.
    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
    # get movie id and title of the top rated movie by the user
    movie_titles = dict(zip(movies['movieId'], movies['title']))
    # find similar movies to the top rated movie
    similar_ids = find_similar_movies(movie_id, X, k)
    movie_title = movie_titles.get(movie_id, "Movie not found")
 
    if movie_title == "Movie not found":
        print(f"Movie with ID {movie_id} not found.")
        return
    
    print(f"Since you watched {movie_title}, you might also like:")
    for i in similar_ids:
        print(movie_titles.get(i, "Movie not found"))


user_id = 150  # Replace with the desired user ID
recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)
