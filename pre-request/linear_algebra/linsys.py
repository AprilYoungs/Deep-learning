from decimal import Decimal, getcontext
from copy import deepcopy

from vector import Vector
from plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def swap_rows(self, row1, row2):
        temp = self.planes[row1]
        self.planes[row1] = self.planes[row2]
        self.planes[row2] = temp


    def multiply_coefficient_and_row(self, coefficient, row):
        try:
            assert coefficient != 0
            p = self.planes[row]
            p.normal_vector = p.normal_vector * coefficient
            p.constant_term = p.constant_term * coefficient
            self.planes[row] = Plane(p.normal_vector,p.constant_term)
        except AssertionError:
            raise Exception('coefficient should be nonzero')


    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        p = self.planes[row_to_add]
        p_added = self.planes[row_to_be_added_to]
        p_added.normal_vector += p.normal_vector * coefficient
        p_added.constant_term += p.constant_term * coefficient
        self.planes[row_to_be_added_to] = Plane(p_added.normal_vector,p_added.constant_term)


    def compute_triangular_form(self):
        """1. swap with topmost row below current row
           2. don't multiply rows by numbers
           3. only add a multiple of a row to rows underneath"""
        system = deepcopy(self)

        # make the first possible triangular
        indices = system.indices_of_first_nonzero_terms_in_each_row()
        i = 0
        while i < len(indices):
            if indices[i] > i and i < len(indices)-1:
                for j in range(i+1,len(indices)):
                    if indices[j] <= i:
                        system.swap_rows(i, j)
                        indices = system.indices_of_first_nonzero_terms_in_each_row()
                        break
            i += 1


        # 消元的算法
        def eliminate(system, index):
            indices = system.indices_of_first_nonzero_terms_in_each_row()
            i = index
            while i < len(indices):
                if indices[i] < i:
                    pi = system[i]

                    # find the first nonzero coefficient
                    for j in range(index-1, i):
                        if not MyDecimal(system[j].normal_vector[indices[i]]).is_near_zero():
                            pib = system[j]
                            coefficient = -pi.normal_vector[indices[i]] / pib.normal_vector[indices[i]]
                            system.add_multiple_times_row_to_row(coefficient, j, i)
                            break
                i += 1

        # 消元
        for i in range(1, len(system.planes)):
            eliminate(system, i)

        return system

    def compute_rref(self):
        """
        Reduced Row-Echelon Form
        *Triangular form
        *Each pivot variable has coefficient 1
        *Each pivot variable is in own column
        :return:
        """
        tf = self.compute_triangular_form()

        indices = tf.indices_of_first_nonzero_terms_in_each_row()
        for i in range(len(indices)-1, 0, -1):
            if indices[i] != -1 and indices[i] == i:
                # 用来消去的项
                pi = tf[i]

                # 找到需要背消的项
                for j in range(i):
                    if not MyDecimal(tf[j].normal_vector[indices[i]]).is_near_zero():
                        pib = tf[j]
                        coefficient = -pib.normal_vector[indices[i]] / pi.normal_vector[indices[i]]
                        tf.add_multiple_times_row_to_row(coefficient, i, j)

        # Each pivot variable has coefficient 1
        indices = tf.indices_of_first_nonzero_terms_in_each_row()
        try:
            for i in range(len(indices)):
                if indices[i] != -1:
                    d = tf[i].normal_vector[indices[i]]
                    tf.multiply_coefficient_and_row(1/d, i)
        except:
            print(tf[i], d)

        return  tf

    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices


    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps


p1 = Plane(normal_vector=Vector([5.862, 1.178, -10.366]), constant_term=-8.15)
p2 = Plane(normal_vector=Vector([-2.931, -0.589, 5.183]), constant_term=-4.075)
s = LinearSystem([p1, p2])
t = s.compute_rref()
print(s)
print('solution {}'.format(t))
print()

p1 = Plane(normal_vector=Vector([8.631, 5.112, -1.816]), constant_term=-5.113)
p2 = Plane(normal_vector=Vector([4.315, 11.132, -5.27]), constant_term=-6.775)
p3 = Plane(normal_vector=Vector([-2.158, 3.01, -1.727]), constant_term=-0.831)
s = LinearSystem([p1, p2, p3])
t = s.compute_rref()
print(s)
print('solution {}'.format(t))
print()

p1 = Plane(normal_vector=Vector(['5.262', '2.739', '-9.878']), constant_term='-3.441')
p2 = Plane(normal_vector=Vector(['5.111', '6.358', '7.638']), constant_term='-2.152')
p3 = Plane(normal_vector=Vector(['2.016', '-9.924', '1.367']), constant_term='-9.278')
p4 = Plane(normal_vector=Vector(['2.167', '-13.593', '18.883']), constant_term='-10.567')
s = LinearSystem([p1, p2, p3, p4])
t = s.compute_rref()
print(s)
print('solution {}'.format(t))

