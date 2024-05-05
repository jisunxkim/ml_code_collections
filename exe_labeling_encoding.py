import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.options.display.max_columns = None 

df = pd.read_csv('./datasets/stats_dataset/census.csv')

# get_dummies works with string columns if they are clean!
 # no need to to change to category for one-hot encoding
pd.get_dummies(df, columns=['race_general', 'sex'], dtype=int, drop_first=True)

cat_cols = ['state_fips_code', 'race_general', 'sex']
df.info()
df[cat_cols] = df[cat_cols].astype('category')

# label encoding method 1
for col in cat_cols:
    df[col+'_cat_num'] = df[col].cat.codes
print(df.head())    
print(list(zip(df['sex'].cat.categories, df['sex'].cat.codes)))
print(list(zip(df['race_general'].cat.categories, df['race_general'].cat.codes)))

# label encoding method 2
encoders = {}
for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col+'_le'] = encoders[col].fit_transform(df[col]) 

print(df.head())

# one-hot encoding
df_onehot = pd.get_dummies(df[cat_cols], drop_first=True, dtype='float')
print(df_onehot.head())
df2 = pd.concat([df, df_onehot], axis='columns')
print(df2.head())

