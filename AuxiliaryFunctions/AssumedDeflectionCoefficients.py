#
from datetime import datetime
from os import mkdir, remove
from os.path import exists, isdir
from sympy import cos, cosh, diff, integrate, simplify, sin, sinh, Symbol
from sympy.solvers.solveset import linsolve
from textwrap import fill

#
A_n = Symbol('A_n')
B_n = Symbol('B_n')
C_n = Symbol('C_n')
D_n = Symbol('D_n')
# gamma = n * pi * lambda / b
gamma_1 = Symbol('gamma_1')
gamma_2 = Symbol('gamma_2')
a = Symbol('a')
x = Symbol('x')
w_p = Symbol('w_p')
dw_p_0 = Symbol('dw_p_0')
dw_p_a = Symbol('dw_p_a')
dddw_p_0 = Symbol('dddw_p_0')
dddw_p_a = Symbol('dddw_p_a')

## Lévy solution

##
#
phi_1 = A_n * cosh(gamma_1 * x) + B_n * sinh(gamma_1 * x) + C_n * cosh(gamma_2 * x) + D_n * sinh(gamma_2 * x)
phi_2 = (A_n + B_n * x) * cosh(gamma_1 * x) + (C_n + D_n * x) * sinh(gamma_1 * x)
phi_3 = (A_n * cos(gamma_2 * x) + B_n * sin(gamma_2 * x) ) * cosh(gamma_1 * x) + (C_n * cos(gamma_2 * x) + D_n * sin(gamma_2 * x) ) * sinh(gamma_1 * x)

#
phi_1_0 = phi_1.subs(x, 0)
phi_2_0 = phi_2.subs(x, 0)
phi_3_0 = phi_3.subs(x, 0)

#
phi_1_a = phi_1.subs(x, a)
phi_2_a = phi_2.subs(x, a)
phi_3_a = phi_3.subs(x, a)


##
#
dphi_1 = diff(phi_1, x)
dphi_2 = diff(phi_2, x)
dphi_3 = diff(phi_3, x)

#
dphi_1_0 = dphi_1.subs(x, 0)
dphi_2_0 = dphi_2.subs(x, 0)
dphi_3_0 = dphi_3.subs(x, 0)

#
dphi_1_a = dphi_1.subs(x, a)
dphi_2_a = dphi_2.subs(x, a)
dphi_3_a = dphi_3.subs(x, a)


##
#
ddphi_1 = diff(dphi_1, x)
ddphi_2 = diff(dphi_2, x)
ddphi_3 = diff(dphi_3, x)

#
ddphi_1_0 = ddphi_1.subs(x, 0)
ddphi_2_0 = ddphi_2.subs(x, 0)
ddphi_3_0 = ddphi_3.subs(x, 0)

#
ddphi_1_a = ddphi_1.subs(x, a)
ddphi_2_a = ddphi_2.subs(x, a)
ddphi_3_a = ddphi_3.subs(x, a)


##
#
dddphi_1 = diff(ddphi_1, x)
dddphi_2 = diff(ddphi_2, x)
dddphi_3 = diff(ddphi_3, x)

#
dddphi_1_0 = dddphi_1.subs(x, 0)
dddphi_2_0 = dddphi_2.subs(x, 0)
dddphi_3_0 = dddphi_3.subs(x, 0)

#
dddphi_1_a = dddphi_1.subs(x, a)
dddphi_2_a = dddphi_2.subs(x, a)
dddphi_3_a = dddphi_3.subs(x, a)


# ##
# # Simply supported/simply supported
# SS_1 = linsolve( [phi_1_0, ddphi_1_0, phi_1_a, ddphi_1_a], (A_n, B_n, C_n, D_n) )
# SS_2 = linsolve( [phi_2_0, ddphi_2_0, phi_2_a, ddphi_2_a], (A_n, B_n, C_n, D_n) )
# SS_3 = linsolve( [phi_3_0, ddphi_3_0, phi_3_a, ddphi_3_a], (A_n, B_n, C_n, D_n) )

# # Clamped/simply supported
# CS_1 = linsolve( [phi_1_0, dphi_1_0 + dw_p_0, phi_1_a, ddphi_1_a], (A_n, B_n, C_n, D_n) )
# CS_2 = linsolve( [phi_2_0, dphi_2_0 + dw_p_0, phi_2_a, ddphi_2_a], (A_n, B_n, C_n, D_n) )
# CS_3 = linsolve( [phi_3_0, dphi_3_0 + dw_p_0, phi_3_a, ddphi_3_a], (A_n, B_n, C_n, D_n) )

# # Free/simply supported
# FS_1 = linsolve( [ddphi_1_0, dddphi_1_0 + dddw_p_0, phi_1_a, ddphi_1_a], (A_n, B_n, C_n, D_n) )
# FS_2 = linsolve( [ddphi_2_0, dddphi_2_0 + dddw_p_0, phi_2_a, ddphi_2_a], (A_n, B_n, C_n, D_n) )
# FS_3 = linsolve( [ddphi_3_0, dddphi_3_0 + dddw_p_0, phi_3_a, ddphi_3_a], (A_n, B_n, C_n, D_n) )

# # Clamped/clamped
# CC_1 = linsolve( [phi_1_0, dphi_1_0 + dw_p_0, phi_1_a, dphi_1_a + dw_p_a], (A_n, B_n, C_n, D_n) )
# CC_2 = linsolve( [phi_2_0, dphi_2_0 + dw_p_0, phi_2_a, dphi_2_a + dw_p_a], (A_n, B_n, C_n, D_n) )
# CC_3 = linsolve( [phi_3_0, dphi_3_0 + dw_p_0, phi_3_a, dphi_3_a + dw_p_a], (A_n, B_n, C_n, D_n) )

# # Free/free
# FF_1 = linsolve( [ddphi_1_0, dddphi_1_0 + dddw_p_0, ddphi_1_a, dddphi_1_a + dddw_p_a], (A_n, B_n, C_n, D_n) )
# FF_2 = linsolve( [ddphi_2_0, dddphi_2_0 + dddw_p_0, ddphi_2_a, dddphi_2_a + dddw_p_a], (A_n, B_n, C_n, D_n) )
# FF_3 = linsolve( [ddphi_3_0, dddphi_3_0 + dddw_p_0, ddphi_3_a, dddphi_3_a + dddw_p_a], (A_n, B_n, C_n, D_n) )

# # Clamped/free
# CF_1 = linsolve( [phi_1_0, dphi_1_0 + dw_p_0, ddphi_1_a, dddphi_1_a + dddw_p_a], (A_n, B_n, C_n, D_n) )
# CF_2 = linsolve( [phi_2_0, dphi_2_0 + dw_p_0, ddphi_2_a, dddphi_2_a + dddw_p_a], (A_n, B_n, C_n, D_n) )
# CF_3 = linsolve( [phi_3_0, dphi_3_0 + dw_p_0, ddphi_3_a, dddphi_3_a + dddw_p_a], (A_n, B_n, C_n, D_n) )

# #
# if not isdir(f'AuxiliaryFunctions/Data'):
#     mkdir(f'AuxiliaryFunctions/Data')

# #
# if exists(f'AuxiliaryFunctions/Data/AssumedDeflectionCoefficients.txt'):
#     remove(f'AuxiliaryFunctions/Data/AssumedDeflectionCoefficients.txt')

# #
# textfile = open(f'AuxiliaryFunctions/Data/AssumedDeflectionCoefficients.txt', 'w', encoding='utf8')

# #
# textfile.write(fill(f'Symbolic expressions for the coefficients of the assumed deflection') )

# #
# textfile.write(f'\n\n')

# #
# textfile.write(fill(f'Source: AssumedDflectionCoefficients.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

# #
# textfile.write(f'\n\n')

# #
# textfile.write(f'Simply supported/simply supported: \n')
# textfile.write(fill(str(SS_1) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(SS_2) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(SS_3) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'Clamped/simply supported: \n')
# textfile.write(fill(str(CS_1) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(CS_2) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(CS_3) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'Free/simply supported: \n')
# textfile.write(fill(str(FS_1) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(FS_2) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(FS_3) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'Clamped/clamped: \n')
# textfile.write(fill(str(CC_1) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(CC_2) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(CC_3) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'Free/free: \n')
# textfile.write(fill(str(FF_1) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(FF_2) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(FF_3) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'Clamped/free: \n')
# textfile.write(fill(str(CF_1) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(CF_2) ) )
# textfile.write(f'\n')
# textfile.write(fill(str(CF_3) ) )
# textfile.write(f'\n\n')

# #
# textfile.close()

##
# Simply supported/simply supported
SS_1 = linsolve( [phi_1_0 + w_p, ddphi_1_0, phi_1_a + w_p, ddphi_1_a], (A_n, B_n, C_n, D_n) )
SS_2 = linsolve( [phi_2_0 + w_p, ddphi_2_0, phi_2_a + w_p, ddphi_2_a], (A_n, B_n, C_n, D_n) )
SS_3 = linsolve( [phi_3_0 + w_p, ddphi_3_0, phi_3_a + w_p, ddphi_3_a], (A_n, B_n, C_n, D_n) )

# Clamped/simply supported
CS_1 = linsolve( [phi_1_0 + w_p, dphi_1_0, phi_1_a + w_p, ddphi_1_a], (A_n, B_n, C_n, D_n) )
CS_2 = linsolve( [phi_2_0 + w_p, dphi_2_0, phi_2_a + w_p, ddphi_2_a], (A_n, B_n, C_n, D_n) )
CS_3 = linsolve( [phi_3_0 + w_p, dphi_3_0, phi_3_a + w_p, ddphi_3_a], (A_n, B_n, C_n, D_n) )

# Free/simply supported
FS_1 = linsolve( [ddphi_1_0, dddphi_1_0, phi_1_a + w_p, ddphi_1_a], (A_n, B_n, C_n, D_n) )
FS_2 = linsolve( [ddphi_2_0, dddphi_2_0, phi_2_a + w_p, ddphi_2_a], (A_n, B_n, C_n, D_n) )
FS_3 = linsolve( [ddphi_3_0, dddphi_3_0, phi_3_a + w_p, ddphi_3_a], (A_n, B_n, C_n, D_n) )

# Clamped/clamped
CC_1 = linsolve( [phi_1_0 + w_p, dphi_1_0, phi_1_a + w_p, dphi_1_a], (A_n, B_n, C_n, D_n) )
CC_2 = linsolve( [phi_2_0 + w_p, dphi_2_0, phi_2_a + w_p, dphi_2_a], (A_n, B_n, C_n, D_n) )
CC_3 = linsolve( [phi_3_0 + w_p, dphi_3_0, phi_3_a + w_p, dphi_3_a], (A_n, B_n, C_n, D_n) )

# Free/free
FF_1 = linsolve( [ddphi_1_0, dddphi_1_0, ddphi_1_a, dddphi_1_a], (A_n, B_n, C_n, D_n) )
FF_2 = linsolve( [ddphi_2_0, dddphi_2_0, ddphi_2_a, dddphi_2_a], (A_n, B_n, C_n, D_n) )
FF_3 = linsolve( [ddphi_3_0, dddphi_3_0, ddphi_3_a, dddphi_3_a], (A_n, B_n, C_n, D_n) )

# Clamped/free
CF_1 = linsolve( [phi_1_0 + w_p, dphi_1_0, ddphi_1_a, dddphi_1_a], (A_n, B_n, C_n, D_n) )
CF_2 = linsolve( [phi_2_0 + w_p, dphi_2_0, ddphi_2_a, dddphi_2_a], (A_n, B_n, C_n, D_n) )
CF_3 = linsolve( [phi_3_0 + w_p, dphi_3_0, ddphi_3_a, dddphi_3_a], (A_n, B_n, C_n, D_n) )

#
if not isdir(f'AuxiliaryFunctions/Data'):
    mkdir(f'AuxiliaryFunctions/Data')

#
if exists(f'AuxiliaryFunctions/Data/AssumedDeflectionCoefficients.txt'):
    remove(f'AuxiliaryFunctions/Data/AssumedDeflectionCoefficients.txt')

#
textfile = open(f'AuxiliaryFunctions/Data/AssumedDeflectionCoefficients.txt', 'w', encoding='utf8')

#
textfile.write(fill(f'Symbolic expressions for the coefficients of the assumed deflection') )

#
textfile.write(f'\n\n')

#
textfile.write(fill(f'Source: AssumedDflectionCoefficients.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

#
textfile.write(f'\n\n')

#
textfile.write(f'Simply supported/simply supported: \n')
textfile.write(fill(str(SS_1) ) )
textfile.write(f'\n')
textfile.write(fill(str(SS_2) ) )
textfile.write(f'\n')
textfile.write(fill(str(SS_3) ) )
textfile.write(f'\n\n')

#
textfile.write(f'Clamped/simply supported: \n')
textfile.write(fill(str(CS_1) ) )
textfile.write(f'\n')
textfile.write(fill(str(CS_2) ) )
textfile.write(f'\n')
textfile.write(fill(str(CS_3) ) )
textfile.write(f'\n\n')

#
textfile.write(f'Free/simply supported: \n')
textfile.write(fill(str(FS_1) ) )
textfile.write(f'\n')
textfile.write(fill(str(FS_2) ) )
textfile.write(f'\n')
textfile.write(fill(str(FS_3) ) )
textfile.write(f'\n\n')

#
textfile.write(f'Clamped/clamped: \n')
textfile.write(fill(str(CC_1) ) )
textfile.write(f'\n')
textfile.write(fill(str(CC_2) ) )
textfile.write(f'\n')
textfile.write(fill(str(CC_3) ) )
textfile.write(f'\n\n')

#
textfile.write(f'Free/free: \n')
textfile.write(fill(str(FF_1) ) )
textfile.write(f'\n')
textfile.write(fill(str(FF_2) ) )
textfile.write(f'\n')
textfile.write(fill(str(FF_3) ) )
textfile.write(f'\n\n')

#
textfile.write(f'Clamped/free: \n')
textfile.write(fill(str(CF_1) ) )
textfile.write(f'\n')
textfile.write(fill(str(CF_2) ) )
textfile.write(f'\n')
textfile.write(fill(str(CF_3) ) )
textfile.write(f'\n\n')

#
textfile.close()
