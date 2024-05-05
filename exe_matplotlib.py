from matplotlib import pyplot as plt  
# x-axis values
x = [5, 2, 9, 4, 7]
# Y-axis values
y = [10, 5, 8, 4, 2]
plt.plot(x, y)
plt.show()
plt.bar(x,y)
plt.show()
plt.hist(y)
plt.show()
plt.scatter(x,y)
plt.show()
plt.plot(x,y, 'ob')
plt.show()
# Data
x = [1, 2, 3, 4, 5]
x2 = [8,10,12,15, 16]
y1, y2 = [10, 20, 15, 25, 30], [5, 15, 10, 20, 25]
plt.fill_between(x, y1, y2, color='skyblue', alpha=0.4, label='Area 1-2')
plt.plot(x, y1, label="line 1", marker='o')
plt.plot(x, y2, label='line 2', marker='o')
plt.plot(x2, y2, label='line 3', marker='o')
plt.show()
