# load data
import pandas as pd
pd.options.display.max_columns = None 

df = pd.read_csv("./datasets/dirty_data/Uncleaned_DS_jobs.csv", index_col='index')
# clean column names
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
#see the data
# any duplicates
df.duplicated().sum()
# remove all duplicates
df = df.drop_duplicates()
# check data
df.info() # most of them are object => need to convert numeric
df.describe()
# investigate individual column
print(df.head(3))
# job_title
len(df.job_title.unique())
# salary_estimate
# $137K-$171K (Glassdoor est.)  
import re
def get_two_numbers(words):
    """
    test_word ='$137K-$171K (Glassdoor est.)'
    """
    words = words.replace('K', '000')
    pattern = r'(\d+)'
    matches = re.findall(pattern=pattern, string=words)
    if not matches:
        matches = [None] * 2
    else:
        matches = [float(v) for v in matches]
        matches += [None] *2
    
    return matches[:2]
   
df[['salary_min', 'salary_max']] = df['salary_estimate'].apply(get_two_numbers).apply(pd.Series)
df[['size_min', 'size_max']] = df['size'].apply(get_two_numbers).apply(pd.Series)
df[['revenue_min', 'revenue_max']] = df['revenue'].apply(get_two_numbers).apply(pd.Series)

df = df.drop(['salary_estimate', 'size'],axis='columns')
df.isnull().sum(axis='index') # missing data in company size
# fill missing data
from sklearn.linear_model import Lasso
model_data = df[['salary_min', 'salary_max', 'rating', 'size_min', 'size_max']].dropna(axis='index')
size_min_lasso_model = Lasso(alpha=0.9)
size_min_lasso_model.fit(X=model_data[['salary_min', 'salary_max', 'rating']], 
                         y=model_data['size_min'])
pred = size_min_lasso_model.predict(model_data[['salary_min', 'salary_max', 'rating']])
actual = model_data['size_min']
def measure_regression_performance(actual, pred):
    from sklearn import metrics 
    from matplotlib import pyplot as plt 
    print(f'R2 Score: {metrics.r2_score(actual, pred):0.3f}')
    print(f'MAE: {metrics.mean_absolute_error(actual, pred):0.3f}')
    plt.scatter(x=actual, y= pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()
measure_regression_performance(actual, pred)
from sklearn.ensemble import RandomForestRegressor
size_min_rf_model = RandomForestRegressor(n_estimators=100)
size_min_rf_model.fit(X=model_data[['salary_min', 'salary_max', 'rating']], 
                y=model_data['size_min'])
pred = size_min_rf_model.predict(model_data[['salary_min', 'salary_max', 'rating']])
actual = model_data['size_min']
measure_regression_performance(actual, pred)
# repalce missing data with the predicted by randomforest
missing_df = df[df.size_min.isna()].copy()
df2 = df.copy()
for index, row in missing_df.iterrows():
    features = row[['salary_min', 'salary_max', 'rating']].values.reshape(1,-1)
    pred = size_min_rf_model.predict(features)
    df2.at[index, 'size_min'] = pred
print(df2.size_min.isna().sum())
# impute by KNN imputer
from sklearn.impute import KNNImputer
impute = KNNImputer()
num_df = df[['salary_min', 'salary_max', 'rating', 'size_min', 'size_max', 'revenue_min', 'revenue_max']].copy()
KNNImputed = impute.fit_transform(num_df)
KNNImputed = pd.DataFrame(KNNImputed, columns = ['salary_min', 'salary_max', 'rating', 'size_min', 'size_max', 'revenue_min', 'revenue_max'])
KNNImputed.isna().sum(axis='index')
df2[['salary_min', 'salary_max', 'rating', 'size_min', 'size_max', 'revenue_min', 'revenue_max']] = KNNImputed
print(df2.isna().sum(axis='index'))
df2['salary_avg'] = (df2.salary_min + df2.salary_max) / 2
df2['revenue_avg'] = (df2.revenue_min + df2.revenue_max) / 2
df2.isna().sum(axis='index')
# there are still 13 rows. We will remove them
df2 = df2.dropna(axis='index')
# standardize numeric column
num_cols = [col for col in df2.columns if df2[col].dtype != 'object']
import numpy as np 
def standardize(arr):
    """
    Z = (x-mu) / sigma
    """
    mu = np.mean(arr)
    sigma = np.std(arr)
    
    return (arr - mu) / sigma 

# this approach doesn't make consistency of the standardizing the data
# as the mean and std are different every data set train, 
# test and actual data for prediction
df2[num_cols].apply(standardize) 
# To ensure consistency in standardizing the data before and 
# after training, you should use 
# the same scaling parameters (mean and standard deviation) 
# for both the training and testing datasets. 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df2.isna().sum(axis='index')
df2.info()
model_data = df2[num_cols].drop(['revenue_min', 'revenue_max'], axis='columns').copy()
target_col = 'revenue_avg'
X_train, X_test, y_train, y_test = train_test_split(
    model_data.drop(target_col, axis='columns'), model_data[target_col],
    train_size=0.6,
    random_state=1)
# fit StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
# scale predictors in training
X_train_scaled = scaler.transform(X_train)
# fit model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
# predict new data
# at this point, scale the new data with the same scaling parameters
# export and import the scaling parameers to use in another system a later time
import joblib 
joblib.dump(scaler, 'scaler.joblib')
# load the scaler and scale new data
loaded_scaler = joblib.load('scaler.joblib')
X_test_scaled = loaded_scaler.transform(X_test)
actual = y_test
pred = lr_model.predict(X=X_test_scaled)
# performance evaluation
measure_regression_performance(actual, pred)
# R2 Score: 0.067
# MAE: 83.748
