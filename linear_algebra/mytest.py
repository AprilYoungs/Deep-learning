import numpy as np

from decimal import *


def swapRows(M, r1, r2):
    for i in range(len(M)):
        temp = M[i][r1-1]
        M[i][r1-1] = M[i][r2-1]
        M[i][r2-1] = temp

def test_swapRows():
    for _ in range(10):
        r, c = np.random.randint(low=1, high=25, size=2)
        matrix = np.random.random((r, c))

        mat = matrix.tolist()

        r1, r2 = np.random.randint(0, r, size=2)
        swapRows(mat, r1, r2)

        matrix[[r1, r2]] = matrix[[r2, r1]]

        self.assertTrue((matrix == np.array(mat)).all(), 'Wrong answer')

test_swapRows()