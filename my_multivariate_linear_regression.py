import pandas as pd
import numpy as np
from my_get_data import get_data
from my_corr_check import check_corr

pd.options.display.max_columns = None
data_url = "./datasets/stats_dataset/mariokart.csv"

if __name__ == "__main__":
    df = get_data(data_url)
    df = df.set_index('id') 
    target = 'total_pr'
    predictors = df.select_dtypes(exclude=['object']).columns.to_list()
    predictors.remove(target)

    X = df[predictors]
    y = df[target]
    
    check_corr(X, y, threshold=None)

    # Multivariate linear regression
    # https://medium.com/analytics-vidhya/implementing-gradient-descent-for-multi-linear-regression-from-scratch-3e31c114ae12
    
    

    
    
               