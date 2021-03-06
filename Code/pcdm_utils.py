import numpy as np
import math
from sympy import *


class Pcdm:
    expr_list = []
    #  (0.955 145 783, 0.022 427 109, 1.836 281 944)
    expr1 = sympify("(x1+x2-1)**2 +x2**2 - (4/(5+(x2+x3-2)**2))+0.4*atan(x1+x2+x3)+2")
    expr_list.append(expr1)

    def __init__(self):
        self.x = []
        self.function = 0
        self.gradient = 0
        self.hessian = 0
        self.A = 0
        self.b = 0
        self.beta = 0
        self.var = {}
        self.size = 0
        self.u = []

    def start(self):
        print("Welcome, This program allows you to find the minimum of a quadratic function using "
              "Powell's Conjugate Direction Method.")
        print("\nDo you want to use 1) one of the following equations 2) your own equation 3) Quit")
        user_input = input("Input 1, 2, or 3: ")
        if user_input == '1':
            self.function = self.assign_function()
            self.run()
        elif user_input == '2':
            self.function = self.create_function()
            self.run()
        else:
            print("Program Complete")

    def create_function(self):
        func = input("Enter your function: ")
        expr = sympify(func)
        user_input = input('Is: {} ok? '.format(expr))
        if user_input.upper() != 'Y' and user_input.upper() != 'YES':
            expr = self.create_function()
        return expr

    def assign_function(self):
        print(len(self.expr_list))
        print("Choose one of the following:")
        for i in range(len(self.expr_list)):
            print(i, ") ", self.expr_list[i])
        valid = False
        while not valid:
            user_input = input("Input number corresponding with equation above: ")
            if len(self.expr_list) > int(user_input) >= 0:
                valid = True
        return self.expr_list[int(user_input)]

    def run(self):
        self.pcdm_initialize()
        print("Commands:\n1) pcdm - run another iteration of Powell's Conjugate Direction Method\n2) Stop")
        while True:
            user_input = input("Enter 1 or 2: ")
            if user_input.upper() == '2':
                break
            self.pcdm_iter2(self.x, self.u)
            # self.b, self.u, self.beta, self.x = self.pcdm_iter(self.A, self.b, self.u, self.beta, self.x, self.size)
        # method here to print results

    def pcdm_initialize(self):
        self.gradient, self.hessian = self.calc_derivatives()
        self.A = (1 / 2) * self.hessian
        self.size = len(self.var)
        self.x = self.starting_point(self.size)
        self.u = self.create_direction_vectors(self.size)
        self.b = [0] * self.size
        self.beta = [0] * self.size

    def starting_point(self, size):
        usr_point = input("Enter initial starting point in form (x,x,x): ")
        point = sympify(usr_point)
        point = np.asarray(point)
        point = np.reshape(point, (size, 1))
        x_lst = [0] * size
        x_lst[0] = point
        return x_lst

    def calc_derivatives(self):
        for x in self.function.free_symbols:
            self.var[str(x)] = sympify(x)
        gradient = derive_by_array(self.function, (self.var['x1'], self.var['x2'], self.var['x3']))
        hessian = derive_by_array(derive_by_array(self.function, (self.var['x1'], self.var['x2'], self.var['x3'])),
                                  (self.var['x1'], self.var['x2'], self.var['x3']))
        return gradient, hessian

    def calc_bi(self, xi):
        return (-1 / 2) * xi

    def create_direction_vectors(self, size):
        u = []
        for i in range(0, size):
            u.append(np.zeros((size, 1)))
            u[i][i] = 1
        return u

    def calc_betai(self, u, b, A, x):
        mat = A.subs('x1', x[0][0]).subs('x2', x[1][0]).subs('x3', x[2][0])
        mat_val = np.array(mat.tolist()).astype(np.float64)
        return np.divide(np.dot(u.transpose(), b), np.dot(np.dot(u.transpose(), mat_val), u))

    def newton(self, x1):
        mat_grad = self.gradient.subs('x1', x1[0]).subs('x2', x1[1]).subs('x3', x1[2])
        mat_grad = np.array(mat_grad.tolist()).astype(np.float64)
        mat_hess = self.hessian.subs('x1', x1[0]).subs('x2', x1[1]).subs('x3', x1[2])
        mat_hess = np.array(mat_hess.tolist()).astype(np.float64)
        val = x1 - mat_grad / mat_hess
        val2 = val.astype(float)
        return np.linalg.det(val2)

    def line_search(self, ff, xx, u, t0):
        t = t0
        x_new = xx + t * u
        # t = self.newton(x_new)
        # x_new = xx + t * u
        # t = self.newton(x_new)
        # x_new = xx + t * u
        # t = self.newton(x_new)
        # x_new = xx + t * u
        return self.newton(x_new)

    def pcdm_iter2(self, x, u):
        for i in range(1, self.size):
            t = self.line_search(self.function, x[i - 1], u[i - 1], 0)
            x[i] = x[i - 1] + t * u[i - 1]
        for i in range(0, self.size - 1):
            u[i] = u[i + 1]
        u[self.size - 1] = x[self.size - 1] - x[0]
        t = self.line_search(self.function, x[0], u[self.size - 1], 0)
        x[0] = x[0] + t * u[self.size - 1]

    def pcdm_iter(self, A, b, u, beta, x, size):
        for i in range(1, size):
            b[i - 1] = self.calc_bi(x[i - 1])
            beta[i - 1] = self.calc_betai(u[i - 1], b[i - 1], A, x[i - 1])
            x[i] = x[i - 1] + np.multiply(b[i - 1], u[i - 1])
        for j in range(0, size - 1):
            u[j] = u[j + 1]
        u[size - 1] = x[size - 1] - x[0]
        b[size - 1] = self.calc_bi(x[0])
        beta[size - 1] = self.calc_betai(u[size - 1], b[size - 1], A, x[0])
        x[0] = x[0] + np.multiply(beta[size - 1], u[size - 1])
        return b, u, beta, x


x = Pcdm()
x.start()
