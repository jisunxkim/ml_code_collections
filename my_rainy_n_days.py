"""
The probability that it will rain tomorrow is dependent on whether or not it is raining today and whether or not it rained yesterday.

If it rained yesterday and today, there is a 20% chance it will rain tomorrow. If it rained one of the days, there is a 60% chance it will rain tomorrow. If it rained neither today nor yesterday, there is a 20% chance it will rain tomorrow.

Given that it is raining today and that it rained yesterday, write a function rain_days to calculate the probability that it will rain on the nth day after today.

Example:

Input:

n=5
Output:

def rain_days(n) -> 0.39968
"""


import numpy as np
from numpy.linalg import matrix_power
def rain_days(n):
    P  = np.zeros((4,4))
    P[0,0] = 0.2
    P[0,1] = 0.8
    P[1,2] = 0.6
    P[1,3] = 0.4
    P[2,0] = 0.6
    P[2,1] = 0.4
    P[3,2] = 0.2
    P[3,3] = 0.8
    P = matrix_power(P,n)
    m = P[0,0] + P[0,2]
    return(m)


