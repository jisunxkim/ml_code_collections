# https://www.geeksforgeeks.org/recommendation-system-in-python/
# content based filtering: Recommend items to the customer similar to previously high-rated items by the customer.
# collaborative filtering: uses a user-item matrix to generate recommendations

import pandas as pd
import numpy as np 

# collaborative filtering
data = [[4, np.nan, np.nan, 5, 1, np.nan, np.nan],
        [5, 5, 4, np.nan, np.nan, 5, np.nan],
        [np.nan, np.nan, np.nan, 2, 4, np.nan, np.nan],
        [np.nan, 3, np.nan, np.nan, np.nan, np.nan, 3]]

df = pd.DataFrame(data, 
                  index = ['u1', 'u2', 'u3', 'u4'],
                  columns=['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7'])

# impute missing rating by users' mean
avg_rows = df.mean(axis='columns')
df_imputed = df.apply(lambda row: row.fillna(avg_rows[row.name]), axis='columns')

# centered rating => normalizing different rating scales of users
centered_rating = df_imputed.apply(lambda row: (row - avg_rows),  axis='index')

# users' similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(centered_rating)

# weighted rating by users' rating and their simiilarity
cos_sim.shape # (4,4) user x user
centered_rating.shape # (4, 7) user x items
sim_weighted_rating =  cos_sim.dot(centered_rating) / cos_sim.sum(axis=0)[:, np.newaxis]
sim_weighted_rating.shape # (4, 7)
pred_rating = (sim_weighted_rating + avg_rows.values[:, np.newaxis]).round(0)
pred_rating
# array([[4., 3., 3., 6., 0., 3., 3.],
#        [5., 5., 4., 5., 5., 5., 5.],
#        [2., 3., 3., 0., 6., 3., 3.],
#        [3., 3., 3., 3., 3., 3., 3.]])
