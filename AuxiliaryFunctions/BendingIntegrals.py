#
from datetime import datetime
from os import mkdir, remove
from os.path import exists, isdir
from sympy import diff, integrate, simplify, Symbol
from textwrap import fill

#
a = Symbol('a')
b = Symbol('b')
x = Symbol('x')
y = Symbol('y')
A = Symbol('A')
B = Symbol('B')
A_11 = Symbol('A_11')
A_12 = Symbol('A_12')
A_16 = Symbol('A_16')
A_22 = Symbol('A_22')
A_26 = Symbol('A_26')
B_11 = Symbol('B_11')
B_12 = Symbol('B_12')
B_16 = Symbol('B_16')
B_21 = Symbol('B_21')
B_22 = Symbol('B_22')
B_26 = Symbol('B_26')

## Ellipse

#
phi = A * (1 - x**2 / a**2 - y**2 / b**2)**2
w   = B * (1 - x**2 / a**2 - y**2 / b**2)**2

#
dphidx = diff(phi, x)
dphidy = diff(phi, y)

#
dwdx = diff(w, x)
dwdy = diff(w, y)

#
ddphiddx = diff(dphidx, x)
ddphiddy = diff(dphidy, y)

#
ddwddx = diff(dwdx, x)
ddwddy = diff(dwdy, y)

# Equation 2.74
u = simplify(
    A_12 * dphidx \
  + A_11 * integrate(ddphiddx, (x, - a * (1 - y**2 / b**2)**(1 / 2), a * (1 - y**2 / b**2)**(1 / 2) ), conds='none') \
  + A_16 * dphidy \
  - B_11 * dwdx \
  - B_12 * integrate(ddwddy, (x, - a * (1 - y**2 / b**2)**(1 / 2), a * (1 - y**2 / b**2)**(1 / 2) ), conds='none') \
  - 2 * B_16 * dwdy)

print(u)

v = simplify(
  - A_26 * dphidx \
  + A_22 * integrate(ddphiddx, (y, - b * (1 - x**2 / a**2)**(1 / 2), b * (1 - x**2 / a**2)**(1 / 2) ), conds='none') \
  + A_12 * dphidy \
  - 2 * B_26 * dwdx \
  - B_21 * integrate(ddwddx, (y, - b * (1 - x**2 / a**2)**(1 / 2), b * (1 - x**2 / a**2)**(1 / 2) ), conds='none') \
  - B_22 * dwdy)

print(v)



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