import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'id': [1, 2, 3, 4, 5, 6],
    'user_id': [1, 2, 1, 3, 4, 5],
    'user_name': ['Jack', 'Alice', 'Jack', 'Scott', 'James', 'Sean'],
    'user_ratings': [894, 311, 233, 852, 922, 579],
    'created_at': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-03', '2022-01-04'],
    'product_id': [101, 102, 103, 104, 105, 106],
    'quantity': [1, 1, 1, 1, 1, 1]
}

def closest_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a table of students and their SAT test scores, write a query to return the two students with the closest test scores with the score difference.

    If there are multiple students with the same minimum score difference, select the student name combination that is higher in the alphabet. 
    """
    # select most recent ratings
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = (
        df
        .sort_values(by='created_at', ascending=False)
        .drop_duplicates(subset=['user_id'], keep='first')
    )
    # make user pairs of different users
    df = (
        df.merge(
            df, 
            how='cross',
            suffixes=('_one', '_other')
            )
    ) 
    df = df[df.user_id_one != df.user_id_other]
    # difference of scores of the user pairs
    df['score_diff'] = abs(df.user_ratings_one - df.user_ratings_other)
    # return the row of the user pair with the smallest difference
    df = (
        df[['user_name_one', 'user_name_other', 'score_diff']]
        .sort_values(by='score_diff', ascending=True)
        )
    
    df.columns = ['one_user', 'other_user', 'score_diff']

    return df.head(1).reset_index(drop=True)

print('the two customers with the closest ratings with the ratings difference: ', closest_ratings(pd.DataFrame(data)), sep='\n')

def count_upsale_custoer(transactions:pd.DataFrame) -> pd.Series:
    """
    count the number of customers of upsale.
    Upsale is to purchase again after the first purchase.
    It doesn't count the purchases of the same day as updsale. 
    """
    transactions['created_at'] = pd.to_datetime(transactions['created_at'])
    transactions_grouped = transactions.groupby('user_id', as_index=False)['created_at'].nunique()
    num_of_upsold_customers = len(transactions_grouped[transactions_grouped.created_at > 1])
    return pd.DataFrame({'num_of_upsold_customers': [num_of_upsold_customers]}) 

# print('number of upsale customers', count_upsale_custoer(pd.DataFrame(data))) 

