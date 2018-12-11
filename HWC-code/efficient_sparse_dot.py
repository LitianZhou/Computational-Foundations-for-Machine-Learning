'''
Dot product of binary sparse vectors
python -m cProfile -s time efficient_sparse_dot.py > out_efficient.txt
'''

import psutil
import os

import numpy as np
myProcess = psutil.Process(os.getpid())

class SparseVector():
    def __init__(self, dimension):
        self.vec = {}
        self.dimension = dimension

    def insert(self, pos_list):
        for each_pos in pos_list:
            self.vec[each_pos] = 1

    def dot_prod(self, b):
        assert(self.dimension == b.dimension)
        return sum([self.vec[each_key]* b.vec[each_key] for each_key in self.vec.keys() if each_key in b.vec.keys()])

DIMENSION = 10000000
a = SparseVector(DIMENSION)
a.insert([1,3,5,7,11])
b = SparseVector(DIMENSION)
b.insert([ind for ind in range(DIMENSION) if ind%3 == 2])


result = a.dot_prod(b)
print(result)

print(myProcess.memory_info())
print(myProcess.memory_percent())