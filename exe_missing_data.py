import pandas as pd
import numpy as np
dict = {'math':[100, 90, np.nan, 95, 100], 
        'english': [30, 45, 56, np.nan, np.nan], 
        'history':[np.nan, 40, 80, 98, 120]} 
df = pd.DataFrame(dict)
print(df)
df.fillna(method='pad') #pad fills with previous one
df.fillna(method='ffill') # same as the pad
df.fillna(method='bfill')
df['math'].fillna(df.math.mean())
df[df['math'].notna()]
df['math'].interpolate()
df.interpolate()

from sklearn.impute import KNNImputer
knnimputer =KNNImputer(n_neighbors=2)
knnimputer.fit(df)
df_imputed = knnimputer.transform(df)
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
print(df_imputed)
print(df)