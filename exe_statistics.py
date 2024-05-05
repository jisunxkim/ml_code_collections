import numpy as np
import pandas as pd
import random
from scipy import stats
import statistics
import statsmodels

import matplotlib.pyplot as plt
# descriptive
# centeral tendency
nums = [random.randint(10, 20) for i in range(100)]
mean = np.mean(nums)
median = np.median(nums)
counts = np.unique(nums, return_counts=True)
mode = np.argmax(counts)
print("using numpy", mean, median, mode)

df = pd.read_csv('./datasets/stats_dataset/airline_delay.csv')
x = df.arr_delay
np.mean(x)
np.median(x.dropna())
np.argmax(np.unique(x, return_counts=True))


# variability
# range, variance, standard deviation
max(x)
min(x)
range = max(x) - min(x)
print(range)
np.var(x)
np.std(x)

statistics.mean(x.dropna())
statistics.median(x)
statistics.median_low(x)
statistics.median_high(x)
statistics.mode(x)
statistics.multimode(x)

# normal distribution
def normal_dist(x, mean, sd):
    """
    Return probability of x from normal distribution 
    """
    pdf_at_x_point = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd)**2)
    return pdf_at_x_point

normal_dist(1, 0, 1)

stats.norm.pdf(x=1, loc=0, scale=1)

values = np.random.normal(loc=0, scale=1, size=10000)
plt.hist(values, bins= 50)
plt.axvline(np.mean(values), color='k', linestyle='dashed')
plt.show()

# x: arrival delayed time
x = x.dropna()
mean = np.mean(x)
sd = np.std(x)
n = len(x)
x1 = 150
z_score = (x1 - mean) / sd
print(x1, '|', z_score)
prob = stats.norm.pdf(x=x1, loc=mean, scale=sd)
c_prob = stats.norm.cdf(x=x1, loc=mean, scale=sd)
print('x: ', x1, 'z-score: ', z_score, 'norm pdf: ', prob, '\nnorm cdf(prob <= x1): ', c_prob)
norm1 = stats.norm(loc = mean, scale=sd)
norm1.pdf(x1)
norm1.cdf(x1)
# percent-point function: norm.ppf(probability_value) returns the z-score corresponding to the specified cumulative probability (probability_value).
norm1.ppf(norm1.cdf(x1))

# Binomial distribution
binomial1 = stats.binom(n=6, p=0.6)
mean, var = binomial1.stats()
print(mean, var)
values = binomial1.rvs(size=100)
plt.hist(values)
plt.show()

# Poisson discrete distributino
avg_num_events_in_interval = 3.5
poisson1 = stats.poisson(mu=avg_num_events_in_interval)
poisson1.pmf(2)
poisson1.cdf(3)
values = poisson1.rvs(size=10000)
plt.hist(values, bins=50)
plt.show()

# Bernoulli distribution
bernoulli1 = stats.bernoulli(p=.2)
bernoulli1.pmf(0)
bernoulli1.cdf(0)
values = bernoulli1.rvs(size=100000)
plt.hist(values, bins=50)
plt.show()

# p-value
# P-value helps us determine how likely it is to get a particular result when the null hypothesis is assumed to be true. It is the probability of getting a sample like ours or more extreme than ours if the null hypothesis is correct. 
# Therefore, if the null hypothesis is assumed to be true, the p-value gives us an estimate of how “strange” our sample is. 
sample_avg_life = 71.8 #years
sample_std = 8.9 #years
n = 100
# Q: will we be able to conclude that the mean life span today is greater than 70 years? here we will use level of significance value of 0.05(alpha). 
#H0: mu = 70
#H1: mu > 70

z_score = (sample_avg_life - 70) / (sample_std/np.sqrt(n))
normal = stats.norm(loc=0, scale = 1)
p_value = normal.cdf(z_score)
print("z_score: ", z_score, "p value: ", p_value)
# as p_value < alpha (5%) => reject H0

# Chi-Square test
data = [[207, 282, 241], [234, 242, 232]]
stat, p, dof, expected = stats.chi2_contingency(data)
print(stat, p, dof, expected)
# as p > 0.5, do not reject HO

# correlation
df.dropna(inplace=True, axis='index')
x = df['arr_delay']
Y = df[['arr_flights', 'weather_ct']]

np.corrcoef(x, Y['arr_flights'])
np.corrcoef(x, Y['weather_ct'])

def pearson_correlation(X,Y):
    if len(X) == len(Y):
        sum_xy = sum((X - np.mean(X))*(Y - np.mean(Y)))
        sum_x_squared = sum((X-np.mean(X))**2)
        sum_y_squared = sum((Y-np.mean(Y))**2)
        corr = sum_xy / np.sqrt(sum_x_squared * sum_y_squared)
    return corr
pearson_correlation(x, Y['arr_flights'])
