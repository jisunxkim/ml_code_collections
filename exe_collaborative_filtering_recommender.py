import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances


pd.options.max_columns = None

# 1. load data, clean, and split
# ratings.dat: UserID::MovieID::Rating::Timestamp
# (1000209, 4)
movies_rating = pd.read_csv(
    './datasets/movie_lens_1m/ratings.dat',
    sep='::',
    header=None, 
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    engine='python'
)
# clean: convert timestamp to datetime
movies_rating['created_at'] = pd.to_datetime(movies_rating['timestamp'], unit='s')
movies_rating.isna().sum(axis='index')
movies_rating.info()

# users.dat: UserID::Gender::Age::Occupation::Zip-code
# 1::F::1::10::48067
movies_users = pd.read_csv(
    './datasets/movie_lens_1m/users.dat',
    sep='::',
    header=None, 
    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
    engine='python'
)
movies_users.shape
movies_users.isna().sum(axis='index')
movies_users['gender'] = movies_users['gender'].astype('category')
movies_users.info()

# 2. Similarities of the users based on their ratings for a particular movie
# objective:
# sample data: sklearn.metrics.pairwise.<methods> cannot handle large data over 50k rows 
# so take only 2% of data
sample_data = movies_rating[['user_id', 'movie_id', 'rating']].sample(
    frac=0.02, replace=False, random_state=1)

# user (row) - movie (column) dataframe
sample_user_movie_matrix = sample_data.pivot(
    index='user_id', columns='movie_id', 
    values='rating')
sample_user_movie_matrix.notna().sum(axis='index')
sample_user_movie_matrix = sample_user_movie_matrix.fillna(0)
sample_user_movie_matrix.shape # (4693, 2704)

# split train and test sets of ratings
train_data, test_data = train_test_split(
    sample_user_movie_matrix, test_size=0.2) # return dataframes

train_data.shape # (3754, 2704)
test_data.shape # (939, 2704)


# pair-wise distance: Pearson correlation coefficient
# metric:
# From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. These metrics support sparse matrix inputs. ['nan_euclidean'] but it does not yet support sparse matrices.
# From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'] 

# user similarity
user_correlation_sim = 1 - pairwise_distances(
    train_data, metric='correlation') # return numpy.ndarray
user_correlation_sim.shape # (3754, 3754)
np.isnan(user_correlation_sim).sum() # 0

user_cosine_sim = 1 - pairwise_distances(train_data, metric='cosine')
user_cosine_sim.shape # (3754, 3754)

# item similarity 
item_correlation_sim = 1 - pairwise_distances(train_data.T, metric='correlation')
item_correlation_sim.shape # (2704, 2704)
np.isnan(item_correlation_sim).sum() # 665406
item_correlation_sim = np.nan_to_num(item_correlation_sim, nan=0)

item_cosine_sim = 1 - pairwise_distances(train_data.T, metric='cosine')
item_cosine_sim.shape # (2704, 2704)

# 3. Predict rating using the similarities among users and items respectively
# test data set
ratings_data = {
    'item1': [0, 1, 2],
    'imte2': [1, 2, 3]
}
ratings_df = pd.DataFrame(ratings_data, index=['user1', 'user2', 'user3'])

similarity_data = np.array([[1, 0.2, 0.3],
                            [0.2, 1, 0.6],
                            [0.3, 0.6, 1]])
similarity_df = pd.DataFrame(similarity_data, index=['user1', 'user2', 'user3'],
                             columns=['user1', 'user2', 'user3'])

def predict_rating(ratings: pd.DataFrame, similarities: np.ndarray, type: str):
    """
    examples of user collaborative filtering
    similarities: 
                user1   user2   user3
        user1   1       0.2     0.3
        user2   0.2     1       0.6
        user3   0.3     0.6     1
    ratings:
                item1 item2
        user1   0       1
        user2   1       2
        user3   2       3
        
    mean_user_rating = mean of the ratings by users:
        user1    0.5
        user2    1.5
        user3    2.5
    Center Ratings: Subtract the mean rating of each user from their respective ratings. This step normalizes the ratings, ensuring that users who rate items on different scales are compared fairly.
    centered_rating = ratings_diff = user-item ratings - mean ratings:
                item1  item2
        user1    -0.5    0.5
        user2    -0.5    0.5
        user3    -0.5    0.5
    weighted_sum_ratings_by_similarities = 
            np.dot(similarities, centered_rating):
        => weighted sum of ratings_dff 
        => take more of centered_rating from the similar user 
                item1   item2 
        user1   -0.75   0.75
        user2   -0.9    0.9 
        user3   -0.95   0.95
    total_user_similarities = sum similarties for each user (index)
        user1    1.5
        user2    1.8
        user3    1.9
    normalized_weighted_sum_ratings_by_similarities = 
            weighted_sum_ratings_by_similarities / total_user_similarities
                item1   item2 
        user1   -0.5    0.5
        user2   -0.5    0.5
        user3   -0.5    0.5
    pred = mean_user_rating + normalized_weighted_sum_ratings_by_similarities
                item1   item2
        user1   0.      1.
        user2   1.      2.
        user3   2.      3.

    """
    # ratings:
    #       movie_id      1         2         3 ... 653.. 866...   3176     
    # user_id           
    #3349               0           0       0       4       1       3
    #933                0           0       0
    if isinstance(ratings, pd.DataFrame):
        ratings_index = ratings.index
        ratings_columns = ratings.columns  
        ratings = ratings.values 
    if isinstance(similarities, pd.DataFrame):
        similarities = similarities.values 
        
    # collaborative filtering based on users
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1).reshape(-1, 1)
        ratings_diff = ratings - mean_user_rating
        weighted_sum_ratings_by_similarities = np.dot(
            similarities, ratings_diff)
        total_user_similarities = np.abs(similarities).sum(axis=1)
        normalized_weighted_sum_ratings_by_similarities =\
            weighted_sum_ratings_by_similarities / total_user_similarities[:,np.newaxis] 
        pred = mean_user_rating + normalized_weighted_sum_ratings_by_similarities    
        # pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    
    elif type == 'item':
        weighted_sum_ratings_by_similarities = np.dot(similarities, ratings)
        total_item_similarities = np.abs(similarities).sum(axis=1)
        pred =  weighted_sum_ratings_by_similarities / total_item_similarities[:, np.newaxis] 
    return pred    


def predict_rating(ratings: pd.DataFrame, similarities: np.ndarray, type: str):
    if isinstance(ratings, pd.DataFrame):
            ratings_index = ratings.index
            ratings_columns = ratings.columns  
            ratings = ratings.values 
    if isinstance(similarities, pd.DataFrame):
        similarities = similarities.values 
        
    # collaborative filtering based on users
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1).reshape(-1, 1)
        ratings_diff = ratings - mean_user_rating
        weighted_sum_ratings_by_similarities = np.dot(
            similarities, ratings_diff)
        total_user_similarities = np.abs(similarities).sum(axis=1)
        normalized_weighted_sum_ratings_by_similarities =\
            weighted_sum_ratings_by_similarities / total_user_similarities[:,np.newaxis] 
        pred = mean_user_rating + normalized_weighted_sum_ratings_by_similarities    
    
    elif type == 'item':
        weighted_sum_ratings_by_similarities = np.dot(similarities, ratings)
        total_item_similarities = np.abs(similarities).sum(axis=1)
        pred =  weighted_sum_ratings_by_similarities / total_item_similarities[:, np.newaxis] 
    return pred    


# predict by user based collaborative filtering
rating_data = train_data.copy()
sim_data = user_correlation_sim.copy()
pred = predict_rating(rating_data, sim_data, 'user')
pred_df_user_based = pd.DataFrame(pred, 
                       index=rating_data.index, 
                       columns=rating_data.columns)

# predict by item based collaborative filtering
rating_data = train_data.T.copy()
sim_data = item_correlation_sim.copy()
sim_data
 
pred = predict_rating(rating_data, sim_data, 'item')
pred_df_item_based = pd.DataFrame(pred, 
                       index=rating_data.index, 
                       columns=rating_data.columns)
pred_df_item_based.shape
pred_df_item_based.head()

# 4. recommend top 5 
def top_rating(df, top_n):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    top_items = {}
    for row, values in df.iterrows():
        top_items[row] = values.nlargest(top_n) # {row_name: pd.Series of values}
    return top_items

top_rating(pred_df_user_based[0:2], 5)

top_rating(pred_df_item_based[0:2], 5)    

