import math
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext


# make the result more precisely
getcontext().prec = 30


class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(Decimal(x) for x in coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

# 让class快遍历
    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if self.num >= len(self.coordinates):
            raise StopIteration
        self.num += 1
        return self.coordinates[self.num-1]

# 让class可以使用下标 indexable
    def __getitem__(self, item):
        return self.coordinates[item]


    def __eq__(self, v):
        if isinstance(v,Vector):
            return self.coordinates == v.coordinates
        #TODO: python exception
        else:
            raise TypeError

    def __add__(self, v):
        assert (len(self.coordinates) == len(v.coordinates))
        coordinate = []
        for i in range(len(v.coordinates)):
            coordinate.append(self.coordinates[i] + v.coordinates[i])
        return Vector(coordinate)

    def __sub__(self, v):
        assert (len(self.coordinates) == len(v.coordinates))
        coordinate = []
        for i in range(len(v.coordinates)):
            coordinate.append(self.coordinates[i] - v.coordinates[i])
        return Vector(coordinate)

    def __mul__(self, s):
        coordinate = []
        for i in range(len(self.coordinates)):
            coordinate.append(self.coordinates[i] * s)
        return Vector(coordinate)

    def __truediv__(self, s):
        coordinate = []
        for i in range(len(self.coordinates)):
            coordinate.append(self.coordinates[i] / s)
        return Vector(coordinate)


    def magnitude(self):
        '''向量长度'''
        muly = 0
        for i in self.coordinates:
            muly += i ** 2
        return muly ** Decimal(0.5)

    def normalized(self):
        '''单位向量'''
        try:
            magnitude = self.magnitude()
            return self / magnitude
        except ZeroDivisionError:
            raise Exception('Cannot normalize the zero ')

    def dot(self, v):
        '''点乘'''
        assert (len(self.coordinates) == len(v.coordinates))
        return sum([x * y for (x, y) in zip(self.coordinates, v.coordinates)])

    def angle_with(self, v, in_degrees=False):
        try:
            assert (len(self.coordinates) == len(v.coordinates))

            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radias = math.acos(u1.dot(u2))

            if in_degrees:
                return math.degrees(angle_in_radias)
            else:
                return angle_in_radias

        except Exception as e:
            raise e

    def parallel(self, v):
        '''平行'''
        assert self.dimension == v.dimension
        return abs(self.dot(v) - self.magnitude()*v.magnitude()) < 1e-10

    def is_orthogonal(self, v):
        '''正交'''
        return abs(self.dot(v)) < 1e-10

    def proj(self, v):
        '''projected on the given vector'''
        uv = v.normalized()
        return uv*self.dot(uv)

    def orthogonalWith(self, v):
        '''return the vector that orthogonal with the projected vector'''
        return self - self.proj(v)

    def cropro(self, v):
        '''cross product'''
        assert(len(self.coordinates)==3)
        assert(len(v.coordinates)==3)
        x1, y1, z1 = self.coordinates
        x2, y2, z2 = v.coordinates
        x3 = y1*z2 - y2*z1
        y3 = -(x1*z2 - x2*z1)
        z3 = x1*y2 - x2*y1
        return Vector([x3, y3, z3])


    def area_of_parallelogram(self, v):
        '''area_of_parallelogram'''
        return self.cropro(v).magnitude()

    def area_of_triangle(self, v):
        '''area_of_triangle'''
        return self.area_of_parallelogram(v)/2
