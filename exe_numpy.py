import numpy as np 
arr = np.array([-1, 2, 5], dtype=np.float32)
print(repr(arr))

# rows: company, columns: departments
# values: number of employee of a department at each company
# Write a function find_percentages to return a five by five matrix that contains the portion of employees employed in each department compared to the total number of employees at each company.
num_employees = np.array( [[10, 20, 30, 30, 10], [15, 15, 5, 10, 5], [150, 50, 100, 150, 50], [300, 200, 300, 100, 100], [1, 5, 1, 1, 2]] )

# sum accross columns (department) for each row (company)
total_employees = num_employees.sum(axis=1) # (5,)
# => array([ 100,   50,  500, 1000,   10])
# NOTE: array transform no effect
# total_employees.T.shape = total_employees.shape = (5,)
# convert shape from (5,) to (5,1)
total_employees = total_employees[:, np.newaxis] #(5,1)
employee_percentages = num_employees / total_employees # (5,5)


a = np.array([0, 1])
b = np.array([9, 8])
c = a
c[0] = 100
print(a)
c = a.copy()
c[0] = 0
print(repr(a))
print(a.dtype)
a = a.astype(np.float32)
print(a.dtype)
arr = np.array([np.nan, np.inf, 2], dtype=np.float32)
print(repr(arr))

# create and shape an array
arr = np.arange(5, dtype=np.float32)
print(repr(arr))
arr = np.arange(-1.5, 4, 2)
print(repr(arr))
arr = np.linspace(5, 11, num=4, endpoint=False, dtype=np.float32)
print(repr(arr))
arr = np.linspace(5, 11, num=4, endpoint=True, dtype=np.float32)
print(repr(arr))
arr = arr.reshape((2,-1))
print(repr(arr))
arr = np.arange(8)
arr = np.reshape(arr, newshape=(4, -1))
print(repr(arr))
arr = np.transpose(arr)
print(repr(arr))
arr = np.transpose(arr, axes=(1, 0))
print(repr(arr))
arr_zeros = np.zeros(4)
arr_ones = np.ones(4)
arr_ones_2d = np.ones(shape=(2,3), dtype=np.float32)
print(repr(arr_zeros), repr(arr_ones), repr(arr_ones_2d), sep='\n')
arr_zeros_2d = np.zeros_like(arr_ones_2d)
print(repr(arr_zeros_2d))

# numpy math
arr1 = np.linspace(start=10, stop=100, num=10, endpoint=True)
print(arr1 + 100)
print(arr1 / 1.5)
print(arr1 // 10)
print(arr1 % 11)
print(arr1 ** 0.5)

def f2c(temp):
    return (5/9)*(temp - 32)

f_temp = np.array([32, -4, 14, -40])
c_temp = f2c(f_temp)
print(f_temp, c_temp, sep='\n')
print(f2c(arr1))
print(f2c(arr1.reshape((2, -1))))
print(np.exp(f_temp))
arr = np.arange(10, 20, 3)
print(np.log2(arr))
print(np.log10(arr))

# matrix calculation
arr1 = np.linspace(1, 100, num=6, endpoint=False,dtype=np.float32)
arr1 = np.reshape(arr1, newshape=(3, 2))
arr2 = np.linspace(10, 100, num=6, endpoint=False, dtype=np.float32)
arr2 = np.reshape(arr2, newshape=(3,2))
print(np.multiply(arr1, arr2))
print(arr1 / arr2)
print(np.divide(arr1, arr2))
print(np.matmul(arr1, arr2.reshape((2,3))))

# random
np.random.seed(123)
rand_arr = np.random.randint(5, 10, size=(2, 5)) # from 5 to 9
print(repr(rand_arr))
np.random.seed(123)
rand_arr2 = np.random.normal(loc=10, scale=2, size=(5, 3))
print(repr(rand_arr2))
vec = np.array([1,2,3])
np.random.shuffle(vec)
print(repr(vec))
arr1 = np.random.binomial(n=10, p=0.7, size=(2,3))
print(repr(arr1))
np.random.shuffle(arr1)
print(repr(arr1))
print(repr(np.random.choice(vec, size=2)))
vec_probabiliity = [0.7, 0.2, 0.1]
print(repr(np.random.choice(vec, size=(3,3), p=vec_probabiliity)))

# index
arr = np.array([[ 1,  2],
                [ 3,  4],
                [ 5,  6],
                [ 7,  8],
                [ 9, 10]])
print(repr(arr), repr(arr[1:]), repr(arr[3:5]), repr(arr[0:2, 1:2]), repr(arr[0,1]),sep='\n')
print(np.argmax(arr)) # 9
print(np.argmin(arr))
print(np.argmax(arr, axis=0)) # [4 4]
print(np.argmax(arr, axis=1)) # [1 1 1 1 1]

# filtering
arr = np.array([[0, 2, np.nan],
                [1, 3, -6],
                [-3, np.nan, 1]])
print(repr(arr == 3))
print(repr(arr > 0))
print(repr(~(arr==1)))
print(repr(np.isnan(arr)))
x, y = np.where(arr > 0)
print(repr(arr[x, y]))
np_filter = [[False, True, False], [True, True, False], [False, False, True]]
print(repr(np.where(np_filter, 1, 0)))
np_filter = np.array([[True, False], [False, True]])
positives = np.array([[1, 2], [3, 4]])
negatives = np.array([[-2, -5], [-1, -8]])
print(repr(np.where(np_filter, positives, negatives))) # array([[ 1, -5],[-1,  4]])
np_filter = positives > 2
print(repr(np.where(np_filter, positives, negatives))) # array([[-2, -5], [ 3,  4]])
print(repr(arr > 0)) # array([[False,  True, False], [ True,  True, False], [False, False,  True]])
print(np.any(arr > 0)) # True
print(np.all(arr > 0)) # False
print(repr(np.any(arr > 1, axis=0))) # array([False,  True, False])
print(repr(np.any(arr > 1, axis=1))) # array([ True,  True, False])

# statistics
arr = np.array([[0, 72, 3, 4], [1, 3, -60, 0], [-3, -2, 4, 10]])
arr.min()
arr.max()
arr.min(axis=0)
arr.min(axis=1)
arr.mean()
np.mean(arr, axis=1)
np.var(arr)
np.median(arr, axis=0)
np.median(arr, axis=-1)
arr2 = np.linspace(1, 100, num= 2*3*4).reshape((2,3,4)).round(0)
print(repr(arr2))
arr2 = np.array([
    [[  1.,   5.,  10.,  14.],
     [ 18.,  23.,  27.,  31.],
     [ 35.,  40.,  44.,  48.]],
    
    [[ 53.,  57.,  61.,  66.],
     [ 70.,  74.,  78.,  83.],
     [ 87.,  91.,  96., 100.]]])
np.median(arr2, axis= -1)

# aggregation
np.sum(arr2)
np.sum(arr2, axis=-1)
np.sum(arr2, axis=0) # (2,3,4) => (3,4)
np.sum(arr2, axis=1) # (2,3,4) => (2,4)
np.sum(arr2, axis=2) # (2,3,4) => (2,3)
np.sum(arr2, axis=(1,2)) # (2,3,4) => (2)

np.cumsum(arr2).shape # without axis it calculate cumsum of the flattened array.
np.cumsum(arr2, axis=0) # (2,3,4) => (2,3,4)
np.cumsum(arr2, axis=1) # (2,3,4) => (2,3,4)
np.cumsum(arr2, axis=2) # (2,3,4) => (2,3,4)

arr1 = np.arange(start=1, stop=(1+ 2*3*4)).reshape(2,3,4)
arr2 = np.arange(start=100, stop=(100+ 2*3*4)).reshape(2,3,4)
print(repr(arr1), repr(arr2), sep='\n')
np.concatenate([arr1, arr2]) # (2,3,4) + (2,3,4) => (4,3,4)
np.concatenate([arr1, arr2], axis=0) #(2,3,4) + (2,3,4) => (4,3,4)
np.concatenate([arr1, arr2], axis=1) #(2,3,4) + (2,3,4) => (4,6,4)
np.concatenate([arr1, arr2], axis=2) # (2,3,4) + (2,3,4) => (2,3,8)

# save and load numpy object
np.save('arr1.npy', arr1)
arr1 = np.load('arr1.npy')
arr1
