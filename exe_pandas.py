# https://www.educative.io/module/lesson/ml-for-software-engineers/B12wp954Yoo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = None 

import pandas as pd
from scipy import sparse

data = {
    'user_id': [1, 2, 3, 4, 5],
    'grade': [10, 10, 11, 10, 11],
    'test score': [85, 60, 90, 30, 99]
}

df = pd.DataFrame(data)
df
#    user_id  grade  test score
# 0        1     10          85
# 1        2     10          60
# 2        3     11          90
# 3        4     10          30
# 4        5     11          99
# the buckets of <50, <75, <90, <100.
bins = [0, 50, 75, 90, 100]
labels = ['<50', '<75', '<90', '<100']

df['test score'] = pd.cut(
    df['test score'], bins=bins, 
    right=False, labels=labels) 

df
#    user_id  grade test score
# 0        1     10        <90
# 1        2     10        <75
# 2        3     11       <100
# 3        4     10        <50
# 4        5     11       <100

def count_normalize_cumsum(df, col):
    df[col] = df[col] / df[col].sum()
    df[col] = df[col].cumsum()*100
    return df


df = (

    df
    .groupby(['grade', 'test score'], observed=False)
    .agg({'user_id': 'count'})
    .reset_index()
    .groupby('grade').apply(lambda df: count_normalize_cumsum(df, 'user_id'))
    .reset_index(drop=True)
)

df['percentage'] = df['user_id'].astype(int).astype(str) + '%'
df[['grade', 'test score', 'percentage']]
#    grade test score percentage
# 0     10        <50        33%
# 1     10        <75        66%
# 2     10        <90       100%
# 3     10       <100       100%
# 4     11        <50         0%
# 5     11        <75         0%
# 6     11        <90         0%
# 7     11       <100       100%





arr = np.array([[0, np.nan, np.nan, 10], [0, 0, 1, 0], [1, 0, 2, 0]])
sparse_df = pd.DataFrame(arr)
sparse_matrix = sparse.lil_matrix(sparse_df)
sparse_matrix.data
# array([list([nan, nan, 10.0]), list([1.0]), list([1.0, 2.0])],
sparse_matrix2 = sparse.csr_matrix(sparse_df)
sparse_matrix2.data
# array([nan, nan, 10.,  1.,  1.,  2.])

data = {'value': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)
df
df.rolling(window=3).sum()

# Generate sample data with different cases
data = {
    'user_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-05', '2024-01-06',
                           '2024-01-07', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12',
                           '2024-01-01', '2024-01-03', '2024-01-04', '2024-01-07', '2024-01-08',
                           '2024-01-09', '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-15']),
    'steps': [1000, 1500, 2000, 1800, 2200, 2500, 2300, 2100, 1900, 1700,
              1200, 1800, 2200, 1600, 2000, 2400, 1700, 2100, 2500, 2300]
}
daily_steps = pd.DataFrame(data)

daily_steps.head()
# Group by user_id and calculate rolling average over 3 consecutive days
# daily_steps['avg_steps'] = 
(
    daily_steps
    .groupby('user_id', as_index=False)
    .rolling(window=3, on='date')
    .sum()
    # .reset_index(level=0, drop=True)
).head()

daily_steps.index = daily_steps['date']
daily_steps.drop('date', axis='columns', inplace=True)
daily_steps.head()
(
    daily_steps
    .groupby('user_id', as_index=False)
    .rolling(window='2D')
    .sum()
    # .reset_index(level=0, drop=True)
).head()

daily_steps.date.is_monotonic_increasing

# Drop rows where rolling average is NaN
avg_table = daily_steps.dropna(subset=['avg_steps'])

print(avg_table[['user_id', 'date', 'avg_steps']])


truck_locations = [
    {"model": "BMW", "location": (1, 2)},
    {"model": "Mercedes", "location": (2, 3)},
    {"model": "Mercedes", "location": (2, 2)},
    {"model": "Mercedes", "location": (2, 3)},
    {"model": "BMW", "location": (1, 2)},
    {"model": "BMW", "location": (3, 3)}]

# Given a list of locations that your trucks are stored at, return the top location (x,y) for each model of truck (Mercedes or BMW).
# truck_locations = {'model': model_name, 'location': (x, y)}
# using pandas
df = pd.DataFrame(truck_locations)
temp = (
    df
    .groupby(['model', 'location'])
    # .size() # return series with index but without series name
    .agg('size')
    .reset_index(name='count_rows')
    # .rename({0: 'count_rows'})
)
temp['max_count'] = temp.groupby(['model'])['count_rows'].transform('max')
temp = temp[temp['count_rows'] == temp['max_count']]
res = {}
for value in temp[['model', 'location']].values:
    res[value[0]] = value[1]
res

# without using pandas
# restructure data: {model: {location: count of the location}}
# using defaultdict
from collections import defaultdict

truck_locations = [
    {"model": "BMW", "location": (1, 2)},
    {"model": "Mercedes", "location": (2, 3)},
    {"model": "Mercedes", "location": (2, 2)},
    {"model": "Mercedes", "location": (2, 3)},
    {"model": "BMW", "location": (1, 2)},
    {"model": "BMW", "location": (3, 3)}]

# Count occurrences of each location for each model
model_location = defaultdict(lambda: defaultdict(int))

for t_dict in truck_locations:
    model = t_dict['model']
    location = tuple(t_dict['location'])
    model_location[model][location] += 1

print(dict(model_location))
# {'BMW': defaultdict(<class 'int'>, {(1, 2): 2, (3, 3): 1}), 'Mercedes': defaultdict(<class 'int'>, {(2, 3): 2, (2, 2): 1})}

# Find locations with the highest frequency for each model
most_frequent_locations = {}
for model, locations in model_location.items():
    max_count = max(locations.values())
    most_frequent_locations[model] = [loc for loc, count in locations.items() if count == max_count]

print(most_frequent_locations)


data = {
    'client_id': [1001, 1001, 1001, 1002, 1002, 1002, 1003, 1003],
    'name': ['James Emerson', 'James Emerson', 'James Emerson',  
             'Fiona Woodward', 'Fiona Woodward', 'Fiona Woodward', 
             'Alvin Gross', 'Alvin Gross'],
    'ranking': [1, 2, 3, 1, 2, 3, 1, 2],
    'value': [1000, np.nan, 1150, np.nan, 1250, np.nan, 1100, 2300]
}

df = pd.DataFrame(data)
df


(
    df['name']
    .str.split(expand=True)
)

(
    df.groupby(['client_id', 'name'], as_index=False)
    .agg(
        ranking_list=('ranking', lambda x: list(x)),
        value_list=('value', lambda x: list(x))
    )
)



(
    df
    .sort_values(['client_id', 'ranking'], ascending=True)
    .groupby('client_id', as_index=False)
    # .sum()
    .fillna(method='ffill')
    )

df['ffill_values'] = (
    df
    .sort_values(['client_id', 'ranking'])
    .groupby('client_id')['value']
    .ffill(limit=1)
)
df['ffill_values'] = df['ffill_values'].ffill()
df
# create 1-D data of objects:  pd.Series
# data of a panda series can be an object
# such as list, set, dictionary, pandas series, pandas dataframe etc.
pd.Series([1,2,3.1], dtype='float64')
pd.Series([[1, 2], [3, 4]])
pd.Series([set([1,2]), set([5,6,7])])
pd.Series([{'1':1, '2': 2}, {'a':1, 'b': 2, 'c': 3}])
series = pd.Series([pd.DataFrame([1,2,3]), pd.DataFrame([[1,2,3], [10,20,30]])])
series[0]
series[0:1]
series = pd.Series({'a':1, 'b':2, 'c':3})
series['b']
(series + 10) ** 0.2
series = pd.Series(data=[1,2,3], index=['a', 'b', 'c'])
series['b']


# create 2-D data: pd.DataFrame
df = pd.DataFrame(
    [[1,100, 'C'], [8, 600, 'A'], [3, 200, 'B'], 
     [2, 400, 'D'], [10, 800,'A'], [7, 700, 'C']],
    index=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'],
    columns=['amount', 'price', 'type']
    )

df.dtypes

# cumulative agg
df['amount'].sum() # 31
df['amount'].cumsum() # [ 1.,  9., 12., 14., 24., 31.]
df.groupby('type')['amount'].sum() # [18,  3,  8,  2] => A 18, B 3, C 8, D 2
df.groupby('type')['amount'].cumsum().values # [ 1,  8,  3,  2, 18,  8]

# concat, merge, drop
df2 = pd.DataFrame([[np.nan, 2]], columns=['c1', 'c2'], index=['r3'])
df3 = pd.concat([df,df2], axis='index')
df3
df2.transpose()
df4 = pd.concat([df.reset_index(drop=True), df2.transpose().reset_index(drop=True)], axis='columns')
df4
df = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4],
                   'c3': [5, 6]},
                  index=['r1', 'r2'])
df.drop(index='r1')
df.drop(labels='r2')
df.drop(columns=['c1'])

df1 = pd.DataFrame({'c1':[1,2], 'c2':[3,4]},
                   index=['r1','r2'])
df2 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]},
                   index=['r1','r2'])
df3 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]})
pd.concat([df1, df2, df3], axis='index')

pd.concat([df1, df2, df3], axis='columns')
pd.concat([df1.reset_index(drop=True), 
           df2.reset_index(drop=True),
           df3.reset_index(drop=True)], axis='columns')

df1 = pd.DataFrame({'name': ['john doe', 'al smith', 'sam black', 'john doe'],
                        'pos': ['1B', 'C', 'P', '2B'],
                        'year': [2000, 2004, 2008, 2003]})
df2 = pd.DataFrame({'name': ['john doe', 'al smith', 'jack lee'],
                        'year': [2000, 2004, 2012],
                        'rbi': [80, 100, 12]})

pd.merge(df1, df2, on=['name', 'year'], how='outer').sort_values(by=['name', 'year'])
pd.merge(df1, df2, on=['name'], how='outer', suffixes=['_df1', '_df2']).sort_values(by=['name', 'year_df1', 'year_df2'])

df = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4],
                   'c3': [5, 6]}, index=['r1', 'r2'])
# dataframe indexing => include:include
# column indexing => can use single or range by :
# row indexing => can use only range by :
col1 = df['c1']
print(type(col1))
col1_df = df[['c1']]
print(type(col1_df))
df['c1'] # column can refer by one column
df['r1': 'r1'] # index cannot refer by one index. WRONG => df['r1'] 
df['r1':'r2'] 
df['r1':]
df[0:1] # index number also works. cannot use single index number WRONG => df[0]
# use iloc[] and loc[] => include:not incdlue
df.iloc[1, 0:1] # second row and first column
df.iloc[1,0] # second row and first column

# file I/O
df = pd.read_csv('./datasets/countries.csv')
df.head(2)
df.dtypes
#clean: convert numeric string cols to numbers
dfolumns = [col.lower().strip().replace(' ', '_') for col in df.columns]
col_object = df.select_dtypes(include='object').columns
df[col_object] = (df[col_object]
    .apply(lambda x: 
            x.str.strip().str.replace(',', '.'), axis='index')
)
col_numeric = df[col_object].dropna().apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all()
col_numeric = col_object[col_numeric]
df[col_numeric] = (
    df[col_numeric]
    .apply(lambda x: pd.to_numeric(x, errors='coerce'))    

)
#write and read
df.to_csv('temp.csv', index='false')
df.to_json('temp.json', index=False)
df = pd.read_json('temp.json')

# group
groups = df.groupby('region')
len(groups)
groups.keys
groups['country'].count()
for name, group in groups:
    print("*"*5, name)
    print(group.iloc[:, :4].head(2))

df_filtered = groups.filter(lambda x: x.name.strip() in ['BALTICS', 'NORTHERN AMERICA'] )

def sum_z_score(arr):
    mu = np.mean(arr)
    sigma = np.std(arr)
    z_score = (arr-mu) / sigma 
    return sum(abs(z_score))

groups.agg(
    total_pop = ('population', 'sum'),
    avg_net_migration = ('net_migration', 'mean'),
    sum_z_score_net_migration = ('net_migration', sum_z_score)
)
# filtering: return True or False each element
filtering1 = df['country'].str.startswith('C')
df[filtering1].iloc[0:2, :4]
filtering2 = df['country'].str.contains('ra')
df[filtering2].iloc[0:2, :4]
df[filtering1 & filtering2].iloc[:5, :4]
df[filtering1 | filtering2].iloc[:5, :4]
filtering3 = df['region'].isin(['EASTERN EUROPE', 'NEAR EAST'])
df[filtering3].iloc[:2, :4]
filtering4 = df['net_migration'].isna()
df[filtering4].iloc[:5, :].loc[:, ['country', 'region', 'population', 'net_migration']]
# NOTE: should put each condition in paranthesis, otherwise behaves unexpected way!!!!
filtering5 = (
    (df['birthrate'] > 30) & 
    (
        df['infant_mortality_(per_1000_births)'] >
        (df['infant_mortality_(per_1000_births)'].mean() + 
        df['infant_mortality_(per_1000_births)'].std()*2 )
        )
    ) 
df[filtering5].groupby('region')['country'].count()
df[filtering5][['region', 'country', 'birthrate', 'infant_mortality_(per_1000_births)']]

# sorting
(
    df[['country', 'region', 'population', 'net_migration']]
    .sort_values(by='population', ascending=False)
    .iloc[:3, :]
)
# metrics of numeric columns
metrics1 = df.describe() # return pandas.DataFrame with indeces of 
#Index(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype='object')
metrics1.loc[['min', '25%', 'max'], ['population', 'net_migration', 'birthrate']]
df[['population', 'net_migration', 'birthrate']].describe(percentiles=[.2, .4, .8, .9])

col_object = df.select_dtypes(exclude='number').columns #pd.DataFrame.select_dtypes() returns a dataframe of the selected c olumns
cat_values = df[col_object].apply(lambda x: x.unique()) # return series. Index(['country', 'region'], dtype='object')
cat_values['region'] # return numpy.ndarray
len(cat_values['region'])
df['region'].value_counts()

# plotting
df.plot(kind='scatter', x='birthrate', y='infant_mortality_(per_1000_births)', bin)
plt.title('birth rate vs. infant mortality per 1000 birth')
plt.show()

(
    pd.cut(df['birthrate'], bins=[0, 0.5, 5, 10, 15, 20, 30, 100])
    .value_counts()
    .plot(kind='bar')
)
plt.show()

# to NumPy
df = pd.DataFrame({'name': ['john doe', 'al smith', 'sam black', 'john doe'],
                    'pos': ['1B', 'C', 'P', '2B'],
                    'sex': ['Male', 'Female', 'Male', 'Male'],
                    'year': [2000, 2004, 2008, 2003]})

df_converted = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype='int', ) # default dtype ='bool', 
# return original dataframe after converting sex to sex_Male; 0 or 1.
df_converted.head()
# convert to a numpy array
df_converted.values

