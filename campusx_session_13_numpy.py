# What is numpy?
'''NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, 
various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including 
mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

At the core of the NumPy package, is the ndarray object. This encapsulates n-dimensional arrays of homogeneous data types'''

# Numpy Arrays Vs Python Sequences
'''NumPy arrays have a fixed size at creation, unlike Python lists (which can grow dynamically). Changing the size of an ndarray will create a new array and delete the original.

The elements in a NumPy array are all required to be of the same data type, and thus will be the same size in memory.

NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data. Typically, such operations are executed more efficiently and with less code than is possible using Python’s built-in sequences.

A growing plethora of scientific and mathematical Python-based packages are using NumPy arrays; though these typically support Python-sequence input, they convert such input to NumPy arrays prior to processing, and they often output NumPy arrays.'''


# Creating Numpy Arrays
# np.array
import numpy as np

a = np.array([1,2,3])
print(a)

# 2D and 3D
b = np.array([[1,2,3],[4,5,6]])
print(b)

c = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(c)

# dtype
d = np.array([1,2,3],dtype=float)
print(d)

# np.arange
e = np.arange(1,11,2)
print(e)

# with reshape
f = np.arange(16).reshape(2,2,2,2)
print(f)

# np.ones and np.zeros
g = np.ones((3,4))
print(g)

h = np.zeros((3,4))
print(h)

# np.random
i = np.random.random((3,4))
print(i)

# np.linspace
j = np.linspace(-10,10,10,dtype=int)
print(j)

# np.identity
k = np.identity(3)
print(k)

# Array Attributes

import numpy as np
a1 = np.arange(10,dtype=np.int32)
a2 = np.arange(12,dtype=float).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

print(a1)
print(a2)
print(a3)

#ndim ---> tell about the dimension
print(a2.ndim)

# shape   --> tell about the shape
print(a3.shape)

# size  -->tell about the size of whole array
print(a2.size)

# itemsize  ---> tell about the size of one element
print(a2.itemsize)
print(a3.itemsize)

# dtype  ---> tell about the type of array
print(a1.dtype)
print(a2.dtype)
print(a3.dtype)

# Changing Datatype
# astype
a = a3.astype(np.int32)
print(a.dtype)

# Array Operations
a1 = np.arange(12).reshape(3,4)
a2 = np.arange(12,24).reshape(3,4)

print(a2)
print(a1)

# scalar operations

# arithmetic
print(a1 ** 2)

# relational
print(a2 == 15)

# vector operations
# arithmetic
print(a1 ** a2)

# Array Functions
a1 = np.random.random((3,3))
a1 = np.round(a1*100)
print(a1)

# max/min/sum/prod
# 0 -> col and 1 -> row
print(np.max(a1,axis=0))
print(np.min(a1,axis=0))
print(np.sum(a1,axis=0))
print(np.prod(a1,axis=0))

print(np.max(a1,axis=1))
print(np.min(a1,axis=1))
print(np.sum(a1,axis=1))
print(np.prod(a1,axis=1))

# mean/median/std/var
print(np.mean(a1,axis=1))
print(np.median(a1,axis=1))
print(np.std(a1,axis=1))
print(np.var(a1,axis=1))

print(np.mean(a1,axis=0))
print(np.median(a1,axis=0))
print(np.std(a1,axis=0))
print(np.var(a1,axis=0))

# trigonomoetric functions
print(np.sin(a1))

# dot product
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(12,24).reshape(4,3)

print(np.dot(a2,a3))

# log and exponents
print(np.exp(a1))

# round/floor/ceil
a1 = np.random.random((2,3))*100
print(a1)
print(np.ceil(a1))
print(np.round(a1))
print(np.floor(a1))

# Indexing and Slicing
a1 = np.arange(10)
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

print(a1)
print(a2)
print(a3)

print(a2[1,0])
print(a3[1,0,1])
print(a3[1,1,0])
print(a1[2:5:2])
print(a2[0:2,1::2])
print(a2[::2,1::2])
print(a2[1,::3])
print(a2[0,:])
print(a2[:,2])
print(a2[1:,1:3])

a3 = np.arange(27).reshape(3,3,3)
print(a3)

print(a3[::2,0,::2])
print(a3[2,1:,1:])
print(a3[0,1,:])

# Iterating

print(a1)
for i in a1:
  print(i)

#***
print(a2)
for i in a2:
  print(i)

#***
print(a3)
for i in a3:
  print(i)

for i in np.nditer(a3):
  print(i)

# Reshaping

# Transpose
print(np.transpose(a2))
print(a2.T)

# ravel
print(a3.ravel())

# Stacking

a4 = np.arange(12).reshape(3,4)
a5 = np.arange(12,24).reshape(3,4)
print(a4)
print(a5)

# horizontal stacking
print(np.hstack((a4,a5)))

# Vertical stacking
print(np.vstack((a4,a5)))

# Splitting

# horizontal splitting
print(a4)
print(np.hsplit(a4,2))

# vertical splitting
# print(a4)
# print(np.hsplit(a4,2))
