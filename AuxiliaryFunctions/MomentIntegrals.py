#
from datetime import datetime
from os import mkdir, remove
from os.path import exists, isdir
from sympy import cos, cosh, diff, integrate, simplify, sin, sinh, Symbol, pi, factor, trigsimp
from textwrap import fill

#
gamma_m = Symbol('gamma_m', real = True, positive = True)
gamma_k = Symbol('gamma_k', real = True, positive = True)
lambda_m = Symbol('lambda_m', real = True, positive = True)
lambda_k = Symbol('lambda_k', real = True, positive = True)
a = Symbol('a', real = True, positive = True)
m = Symbol('m', integer = True, positive = True)
k = Symbol('k', integer = True, positive = True)
x = Symbol('x', real = True, positive = True)
M = Symbol('M')
S3 = Symbol('S3')

# The variation of the in-plane, normal force resulting in a moment is (x belongs to - a / 2 to a / 2)
#
# N = ± 2 * N_0 * x / a
#
# where N_0 = ± N{a / 2}. The corresponding moment is given by
#
# M = int N x dx = ± N_0 * a^2 / 6
#
# The force and moment are therefore related according to
#
# N = 12 * M * x / a^3
#
# Transform the coordinates from (- a / 2, a / 2) to (0, a) yields
#
# N = 12 * M * (1 - 2 * x / a) / a**2
N = 12 * (1 - 2 * x / a) / a**2

# #
# # Simply supported
# #

# #
# xm = sin(m * pi * x / a)
# xk = sin(k * pi * x / a)

# #
# XkXm = simplify(integrate(N * xm * xm, (x, 0, a), conds='none') )

# print(XkXm, flush = True)

# #
# # Clamped
# #

# #
# xm = gamma_m * cos(lambda_m * x / a) - gamma_m * cosh(lambda_m * x / a) + sin(lambda_m * x / a) - sinh(lambda_m * x / a)
# xk = gamma_k * cos(lambda_k * x / a) - gamma_k * cosh(lambda_k * x / a) + sin(lambda_k * x / a) - sinh(lambda_k * x / a)

# #
# XkXm = simplify(integrate(N * xk * xm, (x, 0, a), conds='none') )

# print(XkXm, flush = True)

#
# Clamped Simply supported
#

#
xm = gamma_m * cos(lambda_m * x / a) - gamma_m * cosh(lambda_m * x / a) + sin(lambda_m * x / a) - sinh(lambda_m * x / a)
xk = gamma_k * cos(lambda_k * x / a) - gamma_k * cosh(lambda_k * x / a) + sin(lambda_k * x / a) - sinh(lambda_k * x / a)

#
XkXm = simplify(integrate(N * xk * xm, (x, 0, a), conds='none') )

print(XkXm, flush = True)

# #
# # Free
# #

# #
# x1 = 1

# #
# x2 = S3 * (1 - 2 * x / a)

# #
# x3 = cosh(lambda_m * x / a) + cos(lambda_m * x / a) - gamma_m * sinh(lambda_m * x / a) - gamma_m * sin(lambda_m * x / a)

# #
# x4 = cosh(lambda_k * x / a) + cos(lambda_k * x / a) - gamma_k * sinh(lambda_k * x / a) - gamma_k * sin(lambda_k * x / a)

# #
# X1X1 = simplify(integrate(N * x1 * x1, (x, 0, a), conds='none') )

# print(X1X1, flush = True)

# #
# X1X2 = simplify(integrate(N * x1 * x2, (x, 0, a), conds='none') )

# print(X1X2, flush = True)

# #
# X1X3 = simplify(integrate(N * x1 * x3, (x, 0, a), conds='none') )

# print(X1X3, flush = True)

# #
# X2X2 = simplify(integrate(N * x2 * x2, (x, 0, a), conds='none') )

# print(X2X2, flush = True)

# #
# X2X3 = simplify(integrate(N * x2 * x3, (x, 0, a), conds='none') )

# print(X2X3, flush = True)

# #
# X3X3 = simplify(integrate(N * x3 * x3, (x, 0, a), conds='none') )

# print(X3X3, flush = True)

# #
# X3X3 = simplify(integrate(N * x3 * x4, (x, 0, a), conds='none') )

# print(X3X3, flush = True)