####
# Understand basic concepts
# there are three methods
# 1. pandas.DataFrame.plot(): quick and simple
# 2. matplotlib.pyplot: using pyplot methods - simple and easy for one axes
# 3. matplotlib.pyplot: using figure and axes objects - more control with multiple axes
 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_columns = None
file_path = './datasets/stats_dataset/airline_delay.csv'
df = pd.read_csv(
    filepath_or_buffer=file_path
)
# column we are intersted in. 
target_col = 'arr_delay'
# plot.method
plt.scatter(x=df['arr_flights'], y=df[target_col],
            marker='+')
plt.xlabel(xlabel='number of flights in the airport')
plt.ylabel(ylabel=target_col)
plt.title("total time delayed by other factors")
plt.show()
# pd.DataFrame.plot
df.plot(x=target_col, y='arr_flights',kind = 'scatter')
plt.title("total time delayed by other factors")
plt.show()
# figure
# initializing the data
x = np.array([10, 20, 30, 40])
y = np.array([20, 25, 35, 55])

plt.plot(x,y, 'ro')
# specifies the viewport of the axes. 
plt.axis([5, 50 , 5, 60]) #plotting area: x-left, x-right, y-bottom, y-top 
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y, 'g*')
plt.show()

plt.bar(x,y)
plt.show()

x = np.linspace(0, 2, 100)  # Sample data.

# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, x, label='linear')  # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
ax.plot(x, x**3, label='cubic')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.show()

# pyplot-style:
x = np.linspace(0, 2, 100)  # Sample data.

plt.figure(figsize=(5, 2.7), layout='constrained')
plt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.
plt.plot(x, x**2, label='quadratic')  # etc.
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()

# making helper function
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)  # make 4 random data sets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})