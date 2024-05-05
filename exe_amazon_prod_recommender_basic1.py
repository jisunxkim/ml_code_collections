# source:
# Building a Recommender System for Amazon Products with Python
# https://towardsdatascience.com/building-a-recommender-system-for-amazon-products-with-python-8e0010ec772c

import gzip
import json
import pandas as pd
import numpy as np

# load data from a zipped json file. 
# as it is a large data set. load line by line 
# for a certain limit instead of loading all at once. 

def parse_data(path):
    with gzip.open(path, 'r') as f:
        for line in f:
            yield json.loads(line) 

path = './datasets/amazon_products/Clothing_Shoes_and_Jewelry_5.json.gz'

data_loader = parse_data(path)

df = pd.DataFrame()
for i in range(3):
    line = next(data_loader)
    temp = pd.DataFrame(line)
    df = pd.concat([df, temp], axis='index')
df.reset_index(drop=True, inplace=True)
print(df)

