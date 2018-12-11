'''
Dot product of binary sparse vectors
python -m cProfile -s time sparse_dot.py > out.txt
'''

import numpy as np

import psutil
import os
myProcess = psutil.Process(os.getpid())

class SparseVector():
    def __init__(self, dimension):
        self.vec = [0.0] * dimension
        self.dimension = dimension

    def insert(self, pos_list):
        for each_pos in pos_list:
            self.vec[each_pos] = 1

    def dot_prod(self, b):
        assert (self.dimension == b.dimension)
        return sum([self.vec[ind]* b.vec[ind] for ind in range(self.dimension)])

DIMENSION = 10000000
a = SparseVector(DIMENSION)
a.insert([1,3,5,7,11])
b = SparseVector(DIMENSION)
b.insert([ind for ind in range(DIMENSION) if ind%3 == 2])


result = a.dot_prod(b)
print(result)
print(myProcess.memory_info())
print(myProcess.memory_percent())