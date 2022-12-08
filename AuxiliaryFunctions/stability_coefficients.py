#
from datetime import datetime
from os import mkdir, remove
from os.path import exists, isdir
from sympy import cos, cosh, diff, Matrix, pi, simplify, sin, sinh, sqrt, Symbol, det
from sympy.solvers import solve
from textwrap import fill

#
lambda_1 = Symbol('lambda_1')
lambda_2 = Symbol('lambda_2')
gamma = Symbol('gamma')
a = Symbol('a')
b = Symbol('b')
m = Symbol('m')
y = Symbol('y')
alpha = Symbol('alpha')
beta = Symbol('beta')
delta = Symbol('delta')
N = Symbol('N')
x = Symbol('x')

## Lévy solution

#
f_A = cosh(delta * lambda_1 / a * y)
#
f_B = sinh(delta * lambda_1 / a * y)
#
f_C = cos(delta * lambda_2 / a * y)
#
f_D = sin(delta * lambda_2 / a * y)

#
f_A_0 = f_A.subs(y, 0)
#
f_B_0 = f_B.subs(y, 0)
#
f_C_0 = f_C.subs(y, 0)
#
f_D_0 = f_D.subs(y, 0)

#
f_A_b = f_A.subs(y, b)
#
f_B_b = f_B.subs(y, b)
#
f_C_b = f_C.subs(y, b)
#
f_D_b = f_D.subs(y, b)

#
df_A = diff(f_A, y)
#
df_B = diff(f_B, y)
#
df_C = diff(f_C, y)
#
df_D = diff(f_D, y)

#
df_A_0 = df_A.subs(y, 0)
#
df_B_0 = df_B.subs(y, 0)
#
df_C_0 = df_C.subs(y, 0)
#
df_D_0 = df_D.subs(y, 0)

#
df_A_b = df_A.subs(y, b)
#
df_B_b = df_B.subs(y, b)
#
df_C_b = df_C.subs(y, b)
#
df_D_b = df_D.subs(y, b)

#
ddf_A = diff(df_A, y)
#
ddf_B = diff(df_B, y)
#
ddf_C = diff(df_C, y)
#
ddf_D = diff(df_D, y)

#
ddf_A_0 = ddf_A.subs(y, 0)
#
ddf_B_0 = ddf_B.subs(y, 0)
#
ddf_C_0 = ddf_C.subs(y, 0)
#
ddf_D_0 = ddf_D.subs(y, 0)

#
ddf_A_b = ddf_A.subs(y, b)
#
ddf_B_b = ddf_B.subs(y, b)
#
ddf_C_b = ddf_C.subs(y, b)
#
ddf_D_b = ddf_D.subs(y, b)

#
dddf_A = diff(ddf_A, y)
#
dddf_B = diff(ddf_B, y)
#
dddf_C = diff(ddf_C, y)
#
dddf_D = diff(ddf_D, y)

#
dddf_A_0 = dddf_A.subs(y, 0)
#
dddf_B_0 = dddf_B.subs(y, 0)
#
dddf_C_0 = dddf_C.subs(y, 0)
#
dddf_D_0 = dddf_D.subs(y, 0)

#
dddf_A_b = dddf_A.subs(y, b)
#
dddf_B_b = dddf_B.subs(y, b)
#
dddf_C_b = dddf_C.subs(y, b)
#
dddf_D_b = dddf_D.subs(y, b)

# alpha = D_11 * D_22
# beta - D_12 + 2 * D_66
# gamma_x = D_22 * a**2 / pi**2
# gamma_y = D_11 * b**2 / pi**2
# N_0 = (1 + |x|) * m**2 * alpha / gamma_x or (1 + |x|) * m**2 * alpha / gamma_y
gamma_1 = a * m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta)
#
gamma_2 = a * m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta)

# Simply supported/simply supported
simplysupported = simplify(Matrix( [ [   f_A_0,    f_B_0,    f_C_0,    f_D_0], \
                                     [ ddf_A_0,  ddf_B_0,  ddf_C_0,  ddf_D_0], \
                                     [   f_A_b,    f_B_b,    f_C_b,    f_D_b], \
                                     [ ddf_A_b,  ddf_B_b,  ddf_C_b,  ddf_D_b] ] ).det().subs(delta, 1) )
print(simplysupported, flush=True)

#Verification
simplysupported_solution_lambda_1 = solve(simplysupported, lambda_1)
for i in range(len(simplysupported_solution_lambda_1) ):
    equation = (lambda_1 - simplysupported_solution_lambda_1[i] ).subs(lambda_1, gamma_1).subs(lambda_2, gamma_2)
simplysupported_solution_lambda_2 = solve(simplysupported, lambda_2)
for i in range(len(simplysupported_solution_lambda_2) ):
    equation = (lambda_2 - simplysupported_solution_lambda_2[i] ).subs(lambda_1, gamma_1).subs(lambda_2, gamma_2)

# Clamped/simply supported
clampedsimplysupported = simplify(Matrix( [ [   f_A_0,    f_B_0,    f_C_0,    f_D_0], \
                                            [  df_A_0,   df_B_0,   df_C_0,   df_D_0], \
                                            [   f_A_b,    f_B_b,    f_C_b,    f_D_b], \
                                            [ ddf_A_b,  ddf_B_b,  ddf_C_b,  ddf_D_b] ] ).det().subs(delta, 1) )
print(clampedsimplysupported, flush=True)

# Simply supported/free
freesimplysupported = simplify(Matrix( [ [ ddf_A_b,  ddf_B_b,  ddf_C_b,  ddf_D_b], \
                                         [dddf_A_b, dddf_B_b, dddf_C_b, dddf_D_b], \
                                         [   f_A_0,    f_B_0,    f_C_0,    f_D_0], \
                                         [ ddf_A_0,  ddf_B_0,  ddf_C_0,  ddf_D_0] ] ).det() )
print(freesimplysupported, flush=True)

# Clamped/clamped
clampedclamped = simplify(Matrix( [ [   f_A_0,    f_B_0,    f_C_0,    f_D_0], \
                                    [  df_A_0,   df_B_0,   df_C_0,   df_D_0], \
                                    [   f_A_b,    f_B_b,    f_C_b,    f_D_b], \
                                    [  df_A_b,   df_B_b,   df_C_b,   df_D_b] ] ).det().subs(delta, 1) )
print(clampedclamped, flush=True)

# Equatrion 5.96 from [1]
clmapedclamped_verification = (lambda_1 * lambda_2) / a**2 \
                            * (2 * (1 - cos(lambda_2 * b / a) * cosh(lambda_1 * b / a) ) \
                            - ( (lambda_2**2 - lambda_1**2) / (lambda_1 * lambda_2) ) * sin(lambda_2 * b / a) \
                            * sinh(lambda_1 * b / a) )

# print(simplify(clampedclamped - clmapedclamped_verification) )

# Free/free
freefree = simplify(Matrix( [ [ ddf_A_0,  ddf_B_0,  ddf_C_0,  ddf_D_0], \
                              [dddf_A_0, dddf_B_0, dddf_C_0, dddf_D_0], \
                              [ ddf_A_b,  ddf_B_b,  ddf_C_b,  ddf_D_b], \
                              [dddf_A_b, dddf_B_b, dddf_C_b, dddf_D_b] ] ).det().subs(delta, 0) )
print(freefree, flush=True)

# Clamped/free
clampedfree = simplify(Matrix( [ [   f_A_0,    f_B_0,    f_C_0,    f_D_0], \
                                 [  df_A_0,   df_B_0,   df_C_0,   df_D_0], \
                                 [ ddf_A_b,  ddf_B_b,  ddf_C_b,  ddf_D_b], \
                                 [dddf_A_b, dddf_B_b, dddf_C_b, dddf_D_b] ] ).det() )
print(clampedfree, flush=True)
