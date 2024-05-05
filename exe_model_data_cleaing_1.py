import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = 100
file_url = './datasets/dirty_data/food_coded.csv'

#1. check data file load sample data
df = pd.read_csv(file_url, nrows=10)

print(df.shape)
print(df.dtypes)
print(df.head())

# => csv format OK

# 2. load full data
df = pd.read_csv(file_url)
# 3. Data cleaning 
# 3.1 change columns names
# df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
# 3.2 check duplicated
r_duplicated = df.duplicated()
print('num of the duplicated rows', sum(r_duplicated))
# => no duplicates

# 3.3 check and fix data types of each column
# column, datatype and sample
# check object type if they can converted to numeric
for name, ser in df.iloc[:5, :].items():
    if ser.dtypes == 'object':
        print(name, ser.dtypes, ser[np.random.randint(0,5)], sep='|')
# object type number col: 'gpa', 'weight'
# DO NOT 'coerce' to all columns: df = df.apply(pd.to_numeric, errors='coerece')
# 'coerce' will convert all valid non-numeric object to Nan!
# convert column by column if you are sure what you are doing
print(df[['gpa', 'weight']].dtypes)
# check what non numeric value in the columns.
is_alpha = df[['gpa', 'weight']].apply(lambda x: x.str.isalpha())
df[['gpa', 'weight']][is_alpha.any(axis='columns')]

# 'unknown in gpa. therefore safe to use 'coerce'
# coerce force non-numeric string to NaN
df[['gpa', 'weight']] = df[['gpa', 'weight']].apply(pd.to_numeric, errors='coerce') # use 'ignore' just in case. 
print(df[['gpa', 'weight']].dtypes)
# check any candidate for categorical among the numeric col
# if the number of unique values is than 20 consider for categorical variable.
cat_cols = []
for name, ser in df.items():
    if ser.dtypes != 'object' and len(pd.unique(ser)) <= 20:
        print(name, ser.dtypes, len(pd.unique(ser)), ser[np.random.randint(0, 100)], sep='|')
        cat_cols.append(name)

df[cat_cols] = df[cat_cols].astype(dtype='category')
for col in cat_cols:
    print(df[col].cat.categories)

# 3.4 check missing data
n_row, n_col = df.shape
print(n_row, n_col)
columns_with_missing = df.columns[df.isna().any(axis='index')]
print(columns_with_missing, len(columns_with_missing), 
      round(len(columns_with_missing)/n_col, 2))
# investigate missing data
for name, ser in df[columns_with_missing].items():
    print(name, ser.isna().sum(), ser[1], ser.dtypes,sep='|')
     
print('number of missing data each column: \n', df[columns_with_missing].isna().sum(axis='index').sort_values(
ascending=False))
records_with_missing_values = df[df.isna().any(axis='columns')]
print(records_with_missing_values.shape)

# 3.5 Handling missing data of numeric columns
# check dtypes
print(pd.unique(df.dtypes))
#=> [dtype('float64') dtype('int64') dtype('O')]
# check if xxxday should be datetime
df.filter(regex='.*day.*|.*time.*|.*year.*').head()
#=> calories_day  fruit_day  veggies_day are all number of days. So it looks good.
# get numeric columns - method 1 
num_cols = [col for col in df if df[col].dtypes not in ['object', 'category']]
print(num_cols)
num_cols = df.select_dtypes(exclude=['object', 'category']).columns.to_list()
print(num_cols)
# numeric cols with missing
num_cols_with_missing = df[num_cols].columns[df[num_cols].isna().any(axis='index')]
print(num_cols_with_missing)
#=> Two missing numeric columns: gpa, weight
# impute gpa by mean
df['gpa_imputed'] = df['gpa'].fillna(value=df['gpa'].mean())
print(df[df.gpa.isna()][['gpa_imputed', 'gpa']])



# impute weight by more complex
# KNNImputer impute only numeric types
from sklearn.impute import KNNImputer
num_knn_imputer = KNNImputer(n_neighbors=10)
num_knn_imputer.fit(df[['gpa', 'weight']])
df[['gpa_imputed_knn', 'weight_imputed_knn']] = num_knn_imputer.transform(df[['gpa', 'weight']]) 

imputed_cols = df.columns[df.columns.str.contains('gpa|weights')]

df[imputed_cols][df[imputed_cols].isna().any(axis='columns')]

# impute both numeric and categorical using simple impute
from sklearn.impute import SimpleImputer
category_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = df.select_dtypes(include=['category']).columns
category_imputer.fit(df[cat_cols])
df[cat_cols] = category_imputer.transform(df[cat_cols])

