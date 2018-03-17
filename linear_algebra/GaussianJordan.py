
from fractions import Fraction as f
from copy import deepcopy

class Line(object):
    def __init__(self, v):
        self.vector = [f(e) for e in v]

    def __getitem__(self, item):
        return self.vector[item]

    def __str__(self):
        strSel = [str(e) for e in self.vector]
        return ', '.join(strSel)

    def __mul__(self, other):
        return Line([e*other for e in self.vector])

    def __add__(self, other):
        return Line([e+q for e,q in zip(self.vector, other.vector)])

class Martix(object):
    def __init__(self, lines):
        self.lines = lines

    def __getitem__(self, item):
        return self.lines[item]

    def __str__(self):
        strSel = [str(e) for e in self.lines]
        return '\n'.join(strSel)

    def swap(self, i, j):
        temp = self.lines[i]
        self.lines[i] = self.lines[j]
        self.lines[j] = temp

    def mulity(self, row, num):
        self.lines[row] = self.lines[row]*num

    def add(self, row, toRow, coeffient):
        self.lines[toRow] = self.lines[toRow] + self.lines[row]*coeffient

    def rrff(self):
        system = deepcopy(self)

        for i in range(len(system.lines)):
            # 绝对值的最大值 变换
            max = abs(system[i][i])
            maxI = i
            for j in range(i,len(system.lines)):
                if abs(system[j][i]) > max:
                    max = abs(system[j][i])
                    maxI = j
            # 奇异矩阵
            if max == 0:
                continue
                # return None
            # 将绝对值最大值所在行交换到对角线元素所在行
            if maxI != i:
                system.swap(i, maxI)

            # 列c的对角线元素缩放为1
            coeffient = 1/system.lines[i][i]
            system.mulity(i, coeffient)

            # 多次使用第三个行变换，将列c的其他元素消为0
            for j in range(len(system.lines)):
                if j != i and system[j][i] != 0:
                    coeffient = -system[j][i]
                    system.add(i, j, coeffient)

            print("-----------")
            print(system)
            print("-----------")

        return system


# l1 = Line(['0', '9', '8', '1'])
# l2 = Line(['0','-6', '7', '1'])
# l3 = Line(['0','6', '6', '1'])

l1 = Line(['5', '-6', '-7', '1'])
l2 = Line(['9', '-3', '4', '1'])
l3 = Line(['-2','4', '0', '1'])

m = Martix([l1,l2,l3])
print(m)

m.rrff()

