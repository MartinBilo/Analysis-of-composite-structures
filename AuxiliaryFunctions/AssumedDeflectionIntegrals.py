#
from datetime import datetime
from os import mkdir, remove
from os.path import exists, isdir
from sympy import cos, cosh, diff, integrate, simplify, sin, sinh, Symbol
from textwrap import fill

#
gamma_m = Symbol('gamma_m')
gamma_k = Symbol('gamma_k')
lambda_m = Symbol('lambda_m')
lambda_k = Symbol('lambda_k')
x = Symbol('x')
a = Symbol('a')
S3 = Symbol('S3') # sqrt(3)

## ClampedClampedBeamApproximation

#
m = 12 / a**3 * (a - 2 * x)

#
xm = gamma_m * cos(lambda_m * x / a) - gamma_m * cosh(lambda_m * x / a) + sin(lambda_m * x / a) - sinh(lambda_m * x / a)
xk = gamma_k * cos(lambda_k * x / a) - gamma_k * cosh(lambda_k * x / a) + sin(lambda_k * x / a) - sinh(lambda_k * x / a)

#
dxm = diff(xm, x)
dxk = diff(xk, x)

#
ddxm = diff(dxm, x)
ddxk = diff(dxk, x)

## m != k
#
Xm = simplify(integrate(xm, (x, 0, a), conds='none') )

#
XkXm = simplify(integrate(xk * xm, (x, 0, a), conds='none') )

#
MXkXm = simplify(integrate(m * xk * xm, (x, 0, a), conds='none') )

#
XkddXm = simplify(integrate(xk * ddxm, (x, 0, a), conds='none') )

#
dXkdXm = simplify(integrate(dxk * dxm, (x, 0, a), conds='none') )

#
ddXkddXm = simplify(integrate(ddxk * ddxm, (x, 0, a), conds='none') )

#
XkdXm = simplify(integrate(xk * dxm, (x, 0, a), conds='none') )

#
ddXkdXm = simplify(integrate(ddxk * dxm, (x, 0, a), conds='none') )

## m == k
#
XmXm = simplify(integrate(xm * xm, (x, 0, a), conds='none') )

#
MXmXm = simplify(integrate(m * xm * xm, (x, 0, a), conds='none') )

#
XmddXm = simplify(integrate(xm * ddxm, (x, 0, a), conds='none') )

#
dXmdXm = simplify(integrate(dxm * dxm, (x, 0, a), conds='none') )

#
ddXmddXm = simplify(integrate(ddxm * ddxm, (x, 0, a), conds='none') )

#
XmdXm = simplify(integrate(xm * dxm, (x, 0, a), conds='none') )

#
ddXmdXm = simplify(integrate(ddxm * dxm, (x, 0, a), conds='none') )


#
if not isdir(f'AuxiliaryFunctions/Data'):
    mkdir(f'AuxiliaryFunctions/Data')

#
if exists(f'AuxiliaryFunctions/Data/AssumedDeflectionIntegrals_ClampedClamedBeamApproximation.txt'):
    remove(f'AuxiliaryFunctions/Data/AssumedDeflectionIntegrals_ClampedClamedBeamApproximation.txt')

#
textfile = open(f'AuxiliaryFunctions/Data/AssumedDeflectionIntegrals_ClampedClamedBeamApproximation.txt', 'w', encoding='utf8')

#
textfile.write(fill(f'Symbolic expressions for the integrals of the assumed deflection for a clamped clamped beam approximation') )

#
textfile.write(f'\n\n')

#
textfile.write(fill(f'Source: AssumedDflectionIntegrals.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

#
textfile.write(f'\n\n')

#
textfile.write(f'Xm = \n')
textfile.write(fill(str(Xm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'XkXm = \n')
textfile.write(fill(str(XkXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'XmXm = \n')
textfile.write(fill(str(XmXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'MXkXm = \n')
textfile.write(fill(str(MXkXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'MXmXm = \n')
textfile.write(fill(str(MXmXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'XkddXm = \n')
textfile.write(fill(str(XkddXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'XmddXm = \n')
textfile.write(fill(str(XmddXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'dXkdXm = \n')
textfile.write(fill(str(dXkdXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'dXmdXm = \n')
textfile.write(fill(str(dXmdXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'ddXkddXm = \n')
textfile.write(fill(str(ddXkddXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'ddXmddXm = \n')
textfile.write(fill(str(ddXmddXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'XkdXm = \n')
textfile.write(fill(str(XkdXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'XmdXm = \n')
textfile.write(fill(str(XmdXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'ddXkdXm = \n')
textfile.write(fill(str(ddXkdXm) ) )
textfile.write(f'\n\n')

#
textfile.write(f'ddXmdXm = \n')
textfile.write(fill(str(ddXmdXm) ) )
textfile.write(f'\n\n')

#
textfile.close()

# ## ClampedClampedBeamApproximation

# #
# xm = cosh(lambda_m * x / a) + cos(lambda_m * x / a) - gamma_m * sinh(lambda_m * x / a) - gamma_m * sin(lambda_m * x / a)
# xk = cosh(lambda_k * x / a) + cos(lambda_k * x / a) - gamma_k * sinh(lambda_k * x / a) - gamma_k * sin(lambda_k * x / a)

# x1 = 1
# x2 = S3 * (1 - 2 * x / a)

# X2Xm = simplify(integrate(x2 * xm, (x, 0, a), conds='none') )

# #
# dxm = diff(xm, x)
# dxk = diff(xk, x)

# dx2 = diff(x2, x)

# #
# ddxm = diff(dxm, x)
# ddxk = diff(dxk, x)

# ddXm = simplify(integrate(ddxm, (x, 0, a), conds='none') )

# X2ddXm = simplify(integrate(x2 * ddxm, (x, 0, a), conds='none') )

# dX2dXm = simplify(integrate(dx2 * dxm, (x, 0, a), conds='none') )

# dX2ddXm = simplify(integrate(dx2 * ddxm, (x, 0, a), conds='none') )

# dXm = simplify(integrate(dxm, (x, 0, a), conds='none') )

# X2Xk = simplify(integrate(x2 * xk, (x, 0, a), conds='none') )

# X2dXm = simplify(integrate(x2 * dxm, (x, 0, a), conds='none') )

# print(X2dXm)

# ## m != k
# #
# Xm = simplify(integrate(xm, (x, 0, a), conds='none') )

# #
# XkXm = simplify(integrate(xk * xm, (x, 0, a), conds='none') )

# #
# XkddXm = simplify(integrate(xk * ddxm, (x, 0, a), conds='none') )

# #
# dXkdXm = simplify(integrate(dxk * dxm, (x, 0, a), conds='none') )

# #
# ddXkddXm = simplify(integrate(ddxk * ddxm, (x, 0, a), conds='none') )

# #
# XkdXm = simplify(integrate(xk * dxm, (x, 0, a), conds='none') )

# #
# ddXkdXm = simplify(integrate(ddxk * dxm, (x, 0, a), conds='none') )

# ## m == k
# #
# XmXm = simplify(integrate(xm * xm, (x, 0, a), conds='none') )

# #
# XmddXm = simplify(integrate(xm * ddxm, (x, 0, a), conds='none') )

# #
# dXmdXm = simplify(integrate(dxm * dxm, (x, 0, a), conds='none') )

# #
# ddXmddXm = simplify(integrate(ddxm * ddxm, (x, 0, a), conds='none') )

# #
# XmdXm = simplify(integrate(xm * dxm, (x, 0, a), conds='none') )

# #
# ddXmdXm = simplify(integrate(ddxm * dxm, (x, 0, a), conds='none') )


# #
# if not isdir(f'AuxiliaryFunctions/Data'):
#     mkdir(f'AuxiliaryFunctions/Data')

# #
# if exists(f'AuxiliaryFunctions/Data/AssumedDeflectionIntegrals_ClampedClamedBeamApproximation.txt'):
#     remove(f'AuxiliaryFunctions/Data/AssumedDeflectionIntegrals_ClampedClamedBeamApproximation.txt')

# #
# textfile = open(f'AuxiliaryFunctions/Data/AssumedDeflectionIntegrals_ClampedClamedBeamApproximation.txt', 'w', encoding='utf8')

# #
# textfile.write(fill(f'Symbolic expressions for the integrals of the assumed deflection for a clamped clamped beam approximation') )

# #
# textfile.write(f'\n\n')

# #
# textfile.write(fill(f'Source: AssumedDflectionIntegrals.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

# #
# textfile.write(f'\n\n')

# #
# textfile.write(f'Xm = \n')
# textfile.write(fill(str(Xm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'XkXm = \n')
# textfile.write(fill(str(XkXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'XmXm = \n')
# textfile.write(fill(str(XmXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'XkddXm = \n')
# textfile.write(fill(str(XkddXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'XmddXm = \n')
# textfile.write(fill(str(XmddXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'dXkdXm = \n')
# textfile.write(fill(str(dXkdXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'dXmdXm = \n')
# textfile.write(fill(str(dXmdXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'ddXkddXm = \n')
# textfile.write(fill(str(ddXkddXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'ddXmddXm = \n')
# textfile.write(fill(str(ddXmddXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'XkdXm = \n')
# textfile.write(fill(str(XkdXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'XmdXm = \n')
# textfile.write(fill(str(XmdXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'ddXkdXm = \n')
# textfile.write(fill(str(ddXkdXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.write(f'ddXmdXm = \n')
# textfile.write(fill(str(ddXmdXm) ) )
# textfile.write(f'\n\n')

# #
# textfile.close()