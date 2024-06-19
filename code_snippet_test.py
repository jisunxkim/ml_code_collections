import pandas as pd
from typing import List 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


FILE_PATH = "../../datasets/weather.csv"

np.random.randn()

def load_data(FILE_PATH):
    data = pd.read_csv(FILE_PATH)
    
    return data 

def main():
    data = load_data(FILE_PATH)
    stats =  data.describe().loc[["mean", "std", "count"], :].T
    return stats

if __name__ == "__main__":
    """
    Description: explain the module
    """ 
    stats = main()    
    print(stats)
    print(stats.columns)


