import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn import decomposition
from sklearn import linear_model
from sklearn import metrics

pd.options.display.max_columns = None

# 1. Get data and cleaning
df = pd.read_csv('./datasets/stats_dataset/airline_delay.csv')
df.shape
col_target = ['arr_delay']
col_date_features = ['year', 'month']
col_num_features = list(df.select_dtypes(include='number'))
col_num_features = [
    col for col in col_num_features if (col not in col_date_features) and (col not in col_target)]
col_str_features = [
    col for col in df.columns if( col not in col_num_features ) and (col not in col_date_features)]
col_cat_features = ['carrier']
# convert to categorical variables
df[col_cat_features] = df[col_cat_features].astype('category')
df['carrier'].cat.categories
# drop rows if missing in target col
df = df.dropna(subset=col_target, axis='index')
df.shape
# 2. data for modeling: split 
X_train,  X_test, y_train, y_test = train_test_split(
    df.drop(col_target, axis='columns'), df[col_target], test_size=0.2, random_state=123)

# 3. impute missing values
# don't use fitted one with train for test. Should apply fit for each data set.
# check missing
df.select_dtypes(include='object').isna().sum(axis='index')
df.select_dtypes(include='number').isna().sum(axis='index')

knn_imputer = KNNImputer(n_neighbors=10)
X_train[col_num_features] = knn_imputer.fit_transform(X_train[col_num_features])
X_test[col_num_features] = knn_imputer.fit_transform(X_test[col_num_features])

# simple imputer for both categorical and numerical
# mean, median, most frequentm, or constant
simple_imputer_num = SimpleImputer(strategy='mean')
simple_imputer_cat = SimpleImputer(strategy='most_frequent')

data = df.copy()
data[col_num_features] = simple_imputer_num.fit_transform(data[col_num_features])
data[col_str_features] = simple_imputer_cat.fit_transform(data[col_str_features])

# 4. Scaling
# standardize: scaling by z score
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

std_scale = preprocessing.StandardScaler()
std_scale.fit(X_train[col_num_features])
X_train_scaled[col_num_features] = std_scale.transform(
    X_train[col_num_features])
X_test_scaled[col_num_features] = std_scale.transform(
    X_test[col_num_features])

# min max range scaling
X_train_minmax_scaled = X_train.copy()
X_test_minmax_scaled = X_test.copy()

minmax_scale = preprocessing.MinMaxScaler()
minmax_scale.fit(X_train_minmax_scaled[col_num_features])
X_train_minmax_scaled[col_num_features] = minmax_scale.transform(
    X_train_minmax_scaled[col_num_features])
X_test_minmax_scaled[col_num_features] = minmax_scale.transform(
    X_test_minmax_scaled[col_num_features])

# Robust scaling: use median and IQR
# both Standardization and MinMax normazliation are both affected by outliers. 
X_train_robust_scaled = X_train.copy()
X_test_robust_scaled = X_test.copy()

robust_scaler = preprocessing.RobustScaler()
robust_scaler.fit(X_train_robust_scaled[col_num_features])
X_train_robust_scaled[col_num_features] = robust_scaler.transform(X_train_robust_scaled[col_num_features])
X_test_robust_scaled[col_num_features] = robust_scaler.transform(X_test_robust_scaled[col_num_features])

# apply L2 normalization to each row, 
# in order to calculate cosine similarity scores
# the L2 norm of a row is the square root of the sum of squared values for the row. => should not have a missing value.
X_train_normalized = X_train.copy()
X_test_normalized = X_test.copy()

nomalizer = preprocessing.Normalizer()
nomalizer.fit(X_train_normalized[col_num_features])
X_train_normalized[col_num_features] = nomalizer.transform(X_train_normalized[col_num_features])
X_test_normalized[col_num_features] = nomalizer.transform(X_test_normalized[col_num_features])

# compare data processing
arr_flights = pd.concat([X_train[['arr_flights']], X_train_scaled[['arr_flights']], X_train_minmax_scaled[['arr_flights']], X_train_robust_scaled[['arr_flights']], X_train_normalized[['arr_flights']]], axis='columns')
arr_flights.columns = ['af_original', 'af_standardScale', 'af_minmaxScale', 'af_robustScale', 'af_L2Normalizer']
arr_flights.describe()

# 5. PCA: Dimensionality Reduction
# transform new data using already fitted pca (don't re-fit the pca)
# you should build your PCA on your train set. Then you need to set your principal components to transform your points in test set into the same space. This way you can then use train and test set in the same reducted space.
# PCA is affected by scale, so you need to scale the features in your data before applying PCA. 
pca_obj = decomposition.PCA(n_components=5)
pca_obj.fit(X_train_scaled[col_num_features])

X_train_scaled_pca_num_features = pca_obj.transform(
    X_train_scaled[col_num_features])
X_test_scaled_pca_num_features = pca_obj.transform(
    X_test_scaled[col_num_features])

X_train_scaled_pca_num_features.shape
X_test_scaled_pca_num_features.shape

###### EDA ######
# 6. EDA
# correlation between target and features
cor_res = []
for col in col_num_features:
    cor = np.corrcoef(
        y_train.values.flatten(), 
        X_train_scaled[col].values
        )[0][1]
    cor_res.append((col, round(cor, 2)))

cor_res = sorted(cor_res, key=lambda x: x[1], reverse=True)
print("correlation between target and features: \n", cor_res)

# correlation between features
cor_features_res = []
for left in range(len(col_num_features)-1):
    for right in range(left+1, len(col_num_features)):
        col1 = col_num_features[left]
        col2 = col_num_features[right]
        corr = np.corrcoef(
            X_train_scaled[col1].values,
            X_train_scaled[col2].values
        )[0][1]
        cor_features_res.append((col1, col2, round(corr, 2)))
cor_features_res = sorted(
    cor_features_res,
    key=lambda x: x[2], reverse=True)
print("correlation between features: \n", cor_features_res)

###### MODELING #######
# 7. Modeling
# 7.1 Linear Regression
# The simplest form of linear regression is called least squares regression. This strategy produces a regression model, which is a linear combination of the independent variables, that minimizes the sum of squared residuals between the model's predictions and actual values for the dependent variable.
reg = linear_model.LinearRegression()
reg.fit(X_train_scaled[col_num_features], y_train)
pred_arr_delay = reg.predict(X_test_scaled[col_num_features])
# evaluation
r2 = reg.score(X=X_test_scaled[col_num_features], y=y_test)
mse = metrics.mean_squared_error(
    y_pred=pred_arr_delay, y_true=y_test)
print('r2: ', r2, 'mse: ', round(mse, 5))
# plt.scatter(y_test, pred_arr_delay)
# plt.show()

# 7.2 Ridge Regression
# While ordinary least squares regression is a good way to fit a linear model onto a dataset, it relies on the fact that the dataset's features are each independent, i.e. uncorrelated. When many of the dataset features are linearly correlated, e.g. if a dataset has multiple features depicting the same price in different currencies, it makes the least squares regression model highly sensitive to noise in the data.
# For regularization, the goal is to not only minimize the sum of squared residuals, but to do this with coefficients as small as possible. The smaller the coefficients, the less susceptible they are to random noise in the data. The most commonly used form of regularization is ridge regularization. J(θ)=MSE(θ)+α∑θ**2 
# So, a higher value of α means a stronger regularization effect.

reg = linear_model.Ridge(alpha='0.1')
reg.fit(X_train_scaled[col_num_features], y_train)
pred_arr_delay = reg.predict(X_test_scaled[col_num_features])
# evaluation
r2 = reg.score(X=X_test_scaled[col_num_features], y=y_test)
mse = metrics.mean_squared_error(
    y_pred=pred_arr_delay, y_true=y_test)
print('r2: ', r2, 'mse: ', round(mse, 5))

# Ridge with cross validation to find alpha
alphas = [0.1, 0.2, 0.8, 1] # should be greater than 0
reg = linear_model.RidgeCV(alphas=alphas)
reg.fit(X_train_scaled[col_num_features], y_train)
print(reg.best_score_, reg.coef_, reg.alpha_,sep='\n')

# 7.3 Lasso Regression
# LASSO regularization tends to prefer linear models with fewer parameter values. This means that it will likely zero-out some of the weight coefficients. This reduces the number of features that the model is actually dependent on (since some of the coefficients will now be 0), which can be beneficial when some features are completely irrelevant or duplicates of other features.

reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train_scaled[col_num_features], y_train)
pred_arr_delay = reg.predict(X_test_scaled[col_num_features])
# evaluation
r2 = reg.score(X=X_test_scaled[col_num_features], y=y_test)
mse = metrics.mean_squared_error(
    y_pred=pred_arr_delay, y_true=y_test)
print('r2: ', r2, 'mse: ', round(mse, 5))

# 7.4 Bayesian Regression
# Another way to optimize the hyperparameters of a regularized regression model is with Bayesian techniques. 
# In Bayesian statistics, the main idea is to make certain assumptions about the probability distributions of a model's parameters before being fitted on data. These initial distribution assumptions are called priors for the model's parameters. 
# In a Bayesian ridge regression model, there are two hyperparameters to optimize: α and λ. The α hyperparameter serves the same exact purpose as it does for regular ridge regression; namely, it acts as a scaling factor for the penalty term. 
# The λ hyperparameter acts as the precision of the model's weights. Basically, the smaller the λ value, the greater the variance between the individual weight values.
# Both the α and λ hyperparameters have gamma distribution priors, meaning we assume both values come from a gamma probability distribution.
# There's no need to know the specifics of a gamma distribution, other than the fact that it's a probability distribution defined by a shape parameter and scale parameter.
# Specifically, the α hyperparameter has prior:Γ(α1, α2) and 
# the λ hyperparameter has prior: Γ(λ1, λ2)
# where Γ(k, θ) represents a gamma distribution with shape parameter k and scale parameter θ.

reg = linear_model.BayesianRidge()
reg.fit(X_train_scaled[col_num_features], y_train)
pred_arr_delay = reg.predict(X_test_scaled[col_num_features])
# evaluation
r2 = reg.score(X=X_test_scaled[col_num_features], y=y_test)
mse = metrics.mean_squared_error(
    y_pred=pred_arr_delay, y_true=y_test)
print('r2: ', r2, 'mse: ', round(mse, 5))

# 7.5 Logistic Regression
# The logistic regression model, despite its name, is actually a linear model for classification. It is called logistic regression because it performs regression on logits, which then allows us to classify the data based on model probability predictions.

reg = linear_model.LogisticRegression(solver='lbfgs')
temp_y_label = (y_train > y_train.mean()).values.flatten()
reg.fit(X_train_scaled[col_num_features], temp_y_label)
test_y_label = (y_test > y_test.mean()).values.flatten()
pred_arr_delay_label = reg.predict(X_test_scaled[col_num_features])
acc_score = metrics.accuracy_score(
    y_pred=pred_arr_delay_label, y_true=test_y_label)
print('accuracy: ', round(acc_score, 2))

