import pandas as pd
import numpy as np 

pd.options.display.max_columns=20
# Dummy DataFrame
data = {
    'id_x': [1, 1, 2, 2, 3],
    'name': ['Dept A', 'Dept A', 'Dept B', 'Dept B', 'Dept C'],
    'id_y': [101, 102, 201, 202, 301],
    'salary': [50000, 60000, 70000, 80000, 90000],
    'weight': [0.6, 0.4, 0.7, 0.3, 1.0]
}

df = pd.DataFrame(data)

# Using agg with a custom function for weighted average calculation
df_agg = (
    df
    .groupby(['name'], as_index=False)
    .agg
    (
        number_of_employees=('id_y', 'count'),
        sum_weighted_salary=(
            'salary', 
            lambda x: (x * df.loc[x.index, 'weight']).sum()),
        sum_weighted_salary_devide_by_group_weight=(
            'salary', 
            lambda x: (x * df.loc[x.index, 'weight']).sum() / df.loc[x.index, 'weight'].sum()),
        sum_weighted_salary_devide_by_total_weight=(
            'salary', 
            lambda x: (x * df.loc[x.index, 'weight']).sum() / df['weight'].sum()),
        group_weight=('salary', lambda x: df.loc[x.index, 'weight'].sum()),
        total_weight=('salary', lambda x: df['weight'].sum())
    )
    .sort_values(by='name', ascending=True)
    )

print("Result using agg:")
print(df_agg)

input = [
    {
        'key': 'list1',
        'values': [4,5,2,3,4,5,2,3],
    },
    {
        'key': 'list2',
        'values': [1,1,34,12,40,3,9,7],
    }
]

a = input[0]
for k, arr in a.items():
    print(k, arr)
    
k, arr = a.items()
print(k, arr)
a = [1,2,3]
sum(a)
len(a)