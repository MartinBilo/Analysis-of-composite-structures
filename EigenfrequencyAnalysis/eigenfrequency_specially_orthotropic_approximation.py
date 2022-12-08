# Import modules
from datetime import datetime
from numpy    import arange, argsort, einsum, pi, ones, repeat, sqrt, sum, tile, zeros
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List, Tuple

#
def clamped(m : List[int], n : List[int] ) -> Tuple[List[float], List[float], List[float] ]:
    #
    #    y, n
    #     ^
    #     |
    #     +-------+
    #   ^ |   c   |
    # b | |c     c|
    #   v |   c   |
    # ----+-------+--> x, m
    #     |<----->
    #     |    a

    # Coefficients corresponding to simply supported boundary conditions (table 5.10 from [1])
    a1 = (m == 1) * 4.730 \
       + (m != 1) * (m + 0.5) * pi
    a3 = (n == 1) * 4.730 \
       + (n != 1) * (n + 0.5) * pi
    a2 = ( (m == 1) & (n == 1) ) * 151.3 \
       + ( (m == 1) & (n != 1) ) * 12.30 * a3 * (a3 - 2) \
       + ( (m != 1) & (n == 1) ) * 12.30 * a1 * (a1 - 2) \
       + ( (m != 1) & (n != 1) ) * a1 * a3 * (a1 - 2) * (a3 - 2)

    return(a1, a2, a3)


#
def clampedsimplysupportedclamped(m : List[int], n : List[int] ) -> Tuple[List[float], List[float], List[float] ]:
    #
    #  y, n
    #   ^
    #   |
    #   +-------+
    #   |   s   |
    #   |c     c|
    #   |   c   |
    #   +-------+---> x, m
    #

    # Coefficients corresponding to simply supported boundary conditions (table 5.10 from [1])
    a1 = (m == 1) * 4.730 \
       + (m != 1) * (m + 0.5) * pi
    a3 = (n + 0.25) * pi
    a2 = (m == 1) * 12.30 * a3 * (a3 - 1)\
       + (m != 1) * a1 * a3 * (a1 - 2) * (a3 - 1)

    return(a1, a2, a3)


#
def clampedsimplysupported(m : List[int], n : List[int] ) -> Tuple[List[float], List[float], List[float] ]:
    #
    #  y, n
    #   ^
    #   |
    #   +-------+
    #   |   s   |
    #   |c     c|
    #   |   s   |
    #   +-------+---> x, m
    #

    # Coefficients corresponding to simply supported boundary conditions (table 5.10 from [1])
    a1 = (m == 1) * 4.730 \
       + (m != 1) * (m + 0.5) * pi
    a3 = n * pi
    a2 = (m == 1) * 12.30 * n**2 * pi**2 \
       + (m != 1) * n**2 * pi**2 * a1 * (a1 - 2)

    return(a1, a2, a3)


#
def clampedsimplysupportedclampedsimlpysupported(m : List[int], n : List[int] ) -> Tuple[List[float], List[float], List[float] ]:
    #
    #  y, n
    #   ^
    #   |
    #   +-------+
    #   |   s   |
    #   |c     s|
    #   |   c   |
    #   +-------+---> x, m
    #

    # Coefficients corresponding to simply supported boundary conditions (table 5.10 from [1])
    a1 = (m + 0.25) * pi
    a3 = (n + 0.25) * pi
    a2 = a1 * a3 * (a1 - 1) * (a3 - 1)

    return(a1, a2, a3)


#
def clampedsimplysupportedsimplysupported(m : List[int], n : List[int] ) -> Tuple[List[float], List[float], List[float] ]:
    #
    #  y, n
    #   ^
    #   |
    #   +-------+
    #   |   s   |
    #   |c     s|
    #   |   s   |
    #   +-------+---> x, m
    #

    # Coefficients corresponding to simply supported boundary conditions (table 5.10 from [1])
    a1 = (m + 0.25) * pi
    a3 = n * pi
    a2 = n**2 * pi**2 * a1 * (a1 - 1)

    return(a1, a2, a3)


#
def simplysupported(m : List[int], n : List[int] ) -> Tuple[List[float], List[float], List[float] ]:
    #
    #  y, n
    #   ^
    #   |
    #   +-------+
    #   |   s   |
    #   |s     s|
    #   |   s   |
    #   +-------+---> x, m
    #

    # Coefficients corresponding to simply supported boundary conditions (table 5.10 from [1])
    a1 = m * pi
    a3 = n * pi
    a2 = m**2 * n**2 * pi**4

    return(a1, a2, a3)


#
def vector(x : float, y : float, l : int) -> List[float]:

    z         = x * ones(l)
    z[2:-1:2] = y

    return(z)


# The function 'eigenfrequencies_specially_orthotropic_approximation'
def eigenfrequency_specially_orthotropic_approximation(geometry : Dict[str, float], laminate : Dict[str, float or List[float] ],
                                                       material : Dict[str, float or List[float] ], settings : Dict[str, str or float],
                                                       stiffness : Dict[str, List[float] ] ) -> None:

    #
    a = geometry['a']

    #
    b = geometry['b']

    #
    h = laminate['h']

    #
    t = laminate['t']

    #
    theta = laminate['theta']

    #
    rho = material['rho']

    #
    D = stiffness['D']

    #
    mn_max = settings['mn_max']

    #
    simulation = settings['simulation']

    #
    BoundaryConditions = [f'Simply supported',
                          f'Clamped (x = 0); simply supported (x = a; y = 0,b)',
                          f'Simply supported (x = 0,a); clamped (y = 0); simply supported (y = b)',
                          f'Clamped (x = 0,a); simply supported (y = 0,b)',
                          f'Simply supported (x = 0,a); clamped (y = 0,b)',
                          f'Clamped (x = 0); simply supported (x = a); clamped (y = 0,b)',
                          f'Clamped (x = 0,a); clamped (y = 0); simply supported (y = b)',
                          f'Clamped (x = 0; y = 0); simply supported (x = a; y = b)',
                          f'Clamped']

    #
    situations = len(BoundaryConditions)

    # Number of half-waves in the 1-direction
    m = repeat(arange(1, mn_max + 1, 1), mn_max)

    # Number of half-waves in the 2-direction
    n = tile(arange(1, mn_max + 1, 1), mn_max)

    # Density of the laminate
    rho_laminate = sum(t * rho * ones(len(theta) ) ) / h

    #
    l = vector(a, b, situations)

    # Aspect ratio
    R = vector(a / b, b / a, situations)

    #
    D11 = vector(D[0, 0], D[1, 1], situations)

    #
    D22 = vector(D[1, 1], D[0, 0], situations)

    #
    a1 = zeros( (mn_max**2, situations) )
    a2 = zeros( (mn_max**2, situations) )
    a3 = zeros( (mn_max**2, situations) )

    #
    a1[:, 0], a2[:, 0], a3[:, 0] = simplysupported(m, n)
    a1[:, 1], a2[:, 1], a3[:, 1] = clampedsimplysupportedsimplysupported(m, n)
    a1[:, 2], a2[:, 2], a3[:, 2] = clampedsimplysupportedsimplysupported(n, m)
    a1[:, 3], a2[:, 3], a3[:, 3] = clampedsimplysupported(m, n)
    a1[:, 4], a2[:, 4], a3[:, 4] = clampedsimplysupported(n, m)
    a1[:, 5], a2[:, 5], a3[:, 5] = clampedsimplysupportedclamped(m, n)
    a1[:, 6], a2[:, 6], a3[:, 6] = clampedsimplysupportedclamped(n, m)
    a1[:, 7], a2[:, 7], a3[:, 7] = clampedsimplysupportedclampedsimlpysupported(m, n)
    a1[:, 8], a2[:, 8], a3[:, 8] = clamped(m, n)

    #
    omega = zeros( (mn_max**2, 3, situations) )

    # Eigenfrequencies
    omega[:, 0, :] = 1 / (l**2 * sqrt(rho_laminate) ) * sqrt( \
                     einsum('j,ij->ij', D11, a1**4) \
                   + einsum('j,ij->ij', 2 * (D[0,1] + 2 * D[2,2]) * R**2, a2) \
                   + einsum('j,ij->ij', D22 * R**4, a3**4) )

    # Eigenmodes
    omega[:, 1, :]      = einsum('i,j->ij', m, ones(situations) )
    omega[:, 2, :]      = einsum('i,j->ij', n, ones(situations) )
    omega[:, 1, 2:-1:2] = einsum('i,j->ij', n, ones(situations // 3) )
    omega[:, 1, 2:-1:2] = einsum('i,j->ij', m, ones(situations // 3) )

    # Create and open the text file
    textfile = open(f'{simulation}/Data/EigenfrequencyAnalysis/eigenfrequency_specially_orthotropic_approximation.txt', 'w', encoding='utf8')

    # Add a description of the contents
    textfile.write(fill(f'') )

    # Add an empty line
    textfile.write(f'\n\n')

    # Add the source and a timestamp
    textfile.write(fill(f'Source: eigenfrequency_specially_orthotropic_approximation.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d/%m/%Y")}).') )

    # Add an empty line
    textfile.write(f'\n\n')

    #
    rowIDs = arange(1, mn_max**2 + 1, 1).tolist()

    # Add
    for i in range(situations):
        tmp = omega[:, :, i]
        textfile.write(fill(f'{BoundaryConditions[i] }') )
        textfile.write(f'\n')
        textfile.write(tabulate(tmp[argsort(tmp[:, 0])].tolist(), headers=(f'#', f'\u03C9\u2098\u2099 [Hz]', f'm [-]', f'n [-]'),
                                colalign=('right', 'center', 'center', 'center'), floatfmt=('.0f', '.5e', '.0f', '.0f'), showindex = rowIDs) )

        # Add an empty line
        textfile.write(f'\n\n')

    # Close the text file
    textfile.close()

    #
    return