import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1: 
# Instructions:
#Write a function that takes one numeric argument as input. 
#If the number is larger than zero, the function should return 1, otherwise is should return -1.
#The name of the function should be step

# Your code here:
# -----------------------------------------------
def step(x):
    # doesn't control for numeric input, it just doesn't work if it's not a nr
    if x > 0:
        return 1
    else:
        return -1

# -----------------------------------------------

# Task 2:
# Instructions:
#Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
#The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
#The name of the function should be ReLu

# Your code here:
# -----------------------------------------------
def ReLu(arr, cutoff = 0):
    # arr should be a numpy array
    # second argument is a nr, 0 is nothing is input
    arr_copy = np.copy(arr)
    
    arr_copy[arr_copy < cutoff] = cutoff
    
    return arr_copy

# -----------------------------------------------

# Task 3:
# Instructions:
#Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
#The function should start by multiplying the two numpy arrays (matrix multiplication).
#Next, apply the ReLu function from above to the resulting matrix and return the result.
#Name the function neural_net_layer

# Your code here:
# -----------------------------------------------
def neural_net_layer(x, z):
    # x size is (n,p)
    # z size is p
    y = x.dot(z)
    
    return ReLu(y)

# ------------------------------------------