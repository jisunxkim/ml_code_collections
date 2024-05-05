import pandas as pd
import numpy as np

def check_corr(X: pd.DataFrame, y:pd.Series=None, target: str=None, threshold:float=0.6):
    """
    Check Multicollinearity and 
        Correlation between target and predictors
    Arg:
        X: predctors
        y: target or dependent variable
    Return:
        Multicollinearity
        Correlation
    """
    # check correlations among predictors
    res = []
    if y is not None and target is None:
        if type(y) == pd.Series:
            target = y.name
        elif type(y) == pd.DataFrame:
            target = y.columns[0]
        else:
            target = 'target'
        
        
        
    columns = X.columns.to_list()
    for left in range(len(columns) -1):
        for right in range(left+1, len(columns)):
            corr = np.corrcoef(X.iloc[:, left], X.iloc[:, right])
            corr = np.round(corr[0][1], 2)
            if threshold and (np.abs(corr) > threshold): 
                res.append((columns[left], columns[right], corr))
            elif not threshold:
                res.append((columns[left], columns[right], corr))  
    res = sorted(res, key=lambda x: np.abs(x[2]), reverse=True)        
    print("Correlation between predictors: \n", res) 
    
    if y is not None:
        # check linear relationship between target and predictor
        res = []
        for right in range(len(columns)):
            corr = np.corrcoef(y, X.iloc[:, right])
            corr = np.round(corr[0][1], 2)
            res.append((target, columns[right], corr)) 
        res = sorted(res, key=lambda x: np.abs(x[2]), reverse=True)
        print("correlation between target and predictor: \n", res)