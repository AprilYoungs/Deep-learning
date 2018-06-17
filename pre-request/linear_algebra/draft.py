# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
import numpy as np
from decimal import *
from unittest import TestCase

seed = 7

# TODO 返回矩阵的行数和列数
def shape(M):
    line = len(M)
    row = len(M[0])
    for r in M:
        if len(r) != row:
            raise AttributeError('M should be the same row number')
    return line, row

shape([2,34,23,2])

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = round(M[i][j], decPts)

def make_matrix(num, row, column):
    '''生成所有元素都是固定数字的矩阵'''
    M = [0] * row
    for i in range(row):
        M[i] = [num] * column
    return M


# TODO 计算矩阵的转置
def transpose(M):
    # 生成容器
    l, r = shape(M)
    T = make_matrix(0, r, l)

    # 转置矩阵
    for i in range(len(M)):
        for j in range(len(M[i])):
            T[j][i] = M[i][j]

    return T


# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    try:
        # 确定 输入源可以进行运算
        lA, rA = shape(A)
        lB, rB = shape(B)
        if rA != lB:
            raise ValueError('operands could not be broadcast together')

        # 容器
        result = make_matrix(0, lA, rB)

        # 运算
        Bt = transpose(B)
        for i in range(lA):
            for j in range(rB):
                Ai = A[i]
                Bj = Bt[j]
                e = [a * b for (a, b) in zip(Ai, Bj)]
                e = sum(e)
                result[i][j] = e

    except Exception as e:
        if e != ValueError:
            raise e

    return result

# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    if len(A) == len(b):
        T = [A[i]+b[i] for i in range(len(A))]
    else:
        raise Exception("b with length {} can't be added to A with length {}".format(len(b),len(A)))
    return T

# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    temp = M[r1]
    M[r1] = M[r2]
    M[r2] = temp

# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale != 0:
        M[r] = [e*scale for e in M[r]]
    else:
        raise ValueError("scale can't be zero")

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    M[r1] = [M[r2][i]*scale+M[r1][i] for i in range(len(M[r1]))]


def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    def is_near_zero(num):
        return abs(num) < epsilon

    try:
        Ab = augmentMatrix(A, b)
        for i in range(len(Ab)):
            # 绝对值的最大值 变换
            maxNum = abs(Ab[i][i])
            maxI = i
            for j in range(i, len(Ab)):
                if abs(Ab[j][i]) > maxNum:
                    maxNum = abs(Ab[j][i])
                    maxI = j
            # 奇异矩阵
            if is_near_zero(maxNum):
                raise Exception('奇异矩阵')

            # 将绝对值最大值所在行交换到对角线元素所在行
            if maxI != i:
                swapRows(Ab, i, maxI)

            # 列c的对角线元素缩放为1
            coeffient = 1 / Ab[i][i]
            scaleRow(Ab, i, coeffient)

            # 多次使用第三个行变换，将列c的其他元素消为0
            for j in range(len(Ab)):
                if j != i and not is_near_zero(Ab[j][i]):
                    coeffient = - Ab[j][i]
                    addScaledRow(Ab, j, i, coeffient)
        print(Ab)
        result = [[e[-1]] for e in Ab]
        matxRound(result, decPts)

        return result
    except:
        return None

A = [[5, -6, -7],
     [9, -3,  4],
     [-2, 4,  0]]
b = [[1],[1],[1]]

print(gj_Solve(A, b))


def test_gj_Solve():
    for _ in range(9999):
        r = np.random.randint(low=3, high=10)
        A = np.random.randint(low=-10, high=10, size=(r, r))
        b = np.arange(r).reshape((r, 1))

        x = gj_Solve(A.tolist(), b.tolist(), epsilon=1.0e-8)

        if np.linalg.matrix_rank(A) < r:
            TestCase.assertEqual(x, None, "Matrix A is singular")
        else:
            TestCase.assertNotEqual(x, None, "Matrix A is not singular")
            TestCase.assertEqual(np.array(x).shape, (r, 1),
                             "Expected shape({},1), but got shape{}".format(r, np.array(x).shape))
            Ax = np.dot(A, np.array(x))
            loss = np.mean((Ax - b) ** 2)
            TestCase.assertTrue(loss < 0.1, "Bad result.")



test_gj_Solve()

