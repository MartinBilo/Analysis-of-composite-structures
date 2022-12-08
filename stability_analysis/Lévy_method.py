# Import modules
from datetime       import datetime
from numpy          import array, einsum, sqrt, sin, sinh, cos, cosh, pi, linspace, ones, maximum, argmax
from scipy.optimize import fsolve
from tabulate       import tabulate
from textwrap       import fill
from typing         import Dict, List


#
def determinant_simplysupported(x : List[float], m : List[float], b : float, alpha : float, beta : float, gamma : float,
                                delta : float, mode : float = 0) -> List[float]:
    if mode == 1:
        lambda_1 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta) )
        lambda_2 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta) )
    else:
        lambda_1 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta)
        lambda_2 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta)

    # Modified characteristic equation from stability_coefficients.py in auxilliary_functions
    determinant = (lambda_1**2 + lambda_2**2)**2 * sin(b * lambda_2) * sinh(b * lambda_1)
    return(determinant)


#
def determinant_clampedsimplysupported(x : List[float], m : List[float], b : float, alpha : float, beta : float, gamma : float,
                                       delta : float, mode : float = 0) -> List[float]:
    if mode == 1:
        lambda_1 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta) )
        lambda_2 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta) )
    else:
        lambda_1 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta)
        lambda_2 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta)

    # Modified characteristic equation from stability_coefficients.py in auxilliary_functions
    determinant = ( (lambda_1**3 + lambda_1 * lambda_2**2) * sin(b * lambda_2) * cosh(b * lambda_1) \
                  - (lambda_2**3 + lambda_1**2 * lambda_2) * cos(b * lambda_2) * sinh(b * lambda_1) )
    return(determinant)


#
def determinant_simplysupportedfree(x : List[float], m : List[float], b : float, alpha : float, beta : float, gamma : float,
                                    delta : float, mode : float = 0) -> List[float]:
    if mode == 1:
        lambda_1 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta) )
        lambda_2 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta) )
    else:
        lambda_1 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta)
        lambda_2 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta)

    # Modified characteristic equation from stability_coefficients.py in auxilliary_functions
    determinant = delta**7 * lambda_1**2 * lambda_2**2 \
                * ( (lambda_1**3 + lambda_1 * lambda_2**2) * sin(b * delta * lambda_2) * cosh(b * delta * lambda_1) \
                  - (lambda_2**3 + lambda_1**2 * lambda_2) * cos(b * delta * lambda_2) * sinh(b * delta * lambda_1) )
    return(determinant)


#
def determinant_clamped(x : List[float], m : List[float], b : float, alpha : float, beta : float, gamma : float,
                        delta : float, mode : float = 0) -> List[float]:
    if mode == 1:
        lambda_1 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta) )
        lambda_2 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta) )
    else:
        lambda_1 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta)
        lambda_2 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta)

    # Modified characteristic equation from stability_coefficients.py in auxilliary_functions
    determinant = ( (lambda_1**2 - lambda_2**2) * sin(b * lambda_2) * sinh(b * lambda_1) \
                + 2 * lambda_1 * lambda_2 * (1 - cos(b * lambda_2) * cosh(b * lambda_1) ) )
    return(determinant)


#
def determinant_clampedfree(x : List[float], m : List[float], b : float, alpha : float, beta : float, gamma : float,
                            delta : float, mode : float = 0) -> List[float]:
    if mode == 1:
        lambda_1 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta) )
        lambda_2 = einsum('j,i->ij', m, 1 / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta) )
    else:
        lambda_1 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) + beta)
        lambda_2 = m / sqrt(gamma) * sqrt( sqrt(beta**2 + abs(x) * alpha) - beta)

    # Modified characteristic equation from stability_coefficients.py in auxilliary_functions
    determinant = delta**6 * lambda_1 * lambda_2 * (lambda_1**4 + lambda_2**4 \
                + (lambda_1 * lambda_2**3 - lambda_1**3 * lambda_2) * sin(b * delta * lambda_2) * sinh(b * delta * lambda_1) \
                + 2 * lambda_1**2 * lambda_2**2 * cos(b * delta * lambda_2) * cosh(b * delta * lambda_1) )
    return(determinant)


# The function 'stability_analysis'
def Lévy_method(geometry : Dict[str, float], settings : Dict[str, str or float], stiffness : Dict[str, List[float] ] ) -> None:

    # Lévy method

    # The plate dimension (width) in the x-direction
    a = geometry['a']
    # The plate dimension (length) in the y-direction
    b = geometry['b']

    #
    mn_max = settings['mn_max']
    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    # Counteract overflow
    mn_max = min(mn_max, 10)

    #
    boundary_conditions = [f'Simply supported (y = 0,b)',
                           f'Simply supported (x = 0,a)',
                           f'Clamped (y = 0); simply supported (y = b)',
                           f'Clamped (x = 0); simply supported (x = a)',
                           f'Simply supported (y = 0); free (y = b)',
                           f'Simply supported (x = 0); free (x = a)',
                           f'Clamped (y = 0,b)',
                           f'Clamped (x = 0,a)',
                           f'Clamped (y = 0); free (y = b)',
                           f'Clamped (x = 0); free (x = a)',
                           f'Free (y = 0,b)',
                           f'Free (x = 0,a)']

    #
    applied_force = [f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]',
                     f'N\u1D6A  [N/m]',
                     f'N\u1D67  [N/m]']

    #
    alpha = D[0, 0] * D[1, 1]

    #
    beta = D[0, 1] + 2 * D[2, 2]

    #
    gamma_x = D[1, 1] * a**2 / pi**2
    gamma_y = D[0, 0] * b**2 / pi**2

    # Section 6.2.1 from Kassapoglou [2]
    delta = 5 / 12

    #
    sample_points = linspace(0, 5000, num=50000)

    #
    functions = [determinant_simplysupported, determinant_clampedsimplysupported, determinant_simplysupportedfree, determinant_clamped,
                 determinant_clampedfree]

    N_Lévy = ones(len(boundary_conditions) )

    m_Lévy = ones(len(boundary_conditions) ).astype(str)
    n_Lévy = ones(len(boundary_conditions) ).astype(str)

    o = 0

    #
    mn = linspace(1, mn_max, num = mn_max).astype(int)

    # SPEED UP METHOD
    for fn in functions:
        #
        determinant_sampling_x = fn(sample_points, mn, b, alpha, beta, gamma_x, delta, mode = 1)
        #
        determinant_sampling_y = fn(sample_points, mn, a, alpha, beta, gamma_y, delta, mode = 1)

        #
        index_interval_x = maximum(argmax(determinant_sampling_x > 0, axis = 0), argmax(determinant_sampling_x < 0, axis = 0) ).astype(int)

        #
        index_interval_y = maximum(argmax(determinant_sampling_y > 0, axis = 0), argmax(determinant_sampling_y < 0, axis = 0) ).astype(int)

        #
        x0 = sample_points[index_interval_x - 1] - (determinant_sampling_x[index_interval_x - 1, mn - 1] * (sample_points[index_interval_x] - sample_points[index_interval_x - 1] ) ) / (determinant_sampling_x[index_interval_x, mn - 1] - determinant_sampling_x[index_interval_x - 1, mn - 1] )

        #
        y0 = sample_points[index_interval_y - 1] - (determinant_sampling_y[index_interval_y - 1, mn - 1] * (sample_points[index_interval_y] - sample_points[index_interval_y - 1] ) ) / (determinant_sampling_y[index_interval_y, mn - 1] - determinant_sampling_y[index_interval_y - 1, mn - 1] )

        #
        n_x =  - (abs(fsolve(fn, x0, args = (mn, b, alpha, beta, gamma_x, delta) ) ) + 1) * alpha * mn**2 / gamma_x

        #
        n_y =  - (abs(fsolve(fn, y0, args = (mn, a, alpha, beta, gamma_y, delta) ) ) + 1) * alpha * mn**2 / gamma_y

        #
        m_Lévy[2 * o]     = f'{mn[argmax(n_x) ]:.0f}'
        n_Lévy[2 * o]     = f'1'

        #
        m_Lévy[2 * o + 1] = f'1'
        n_Lévy[2 * o + 1] = f'{mn[argmax(n_y) ]:.0f}'

        #
        N_Lévy[2 * o]     = max(n_x)
        N_Lévy[2 * o + 1] = max(n_y)

        #
        o += 1

    # Derived from equation 6.21 in Kassapoglou [2] and equation 5.91 from [1] for f = constant
    N_Lévy[2 * o] = - D[0, 0] * pi**2 / a**2
    N_Lévy[2 * o + 1] = - D[1, 1] * pi**2 / b**2

    #
    m_Lévy[2 * o]     = f'1'
    n_Lévy[2 * o]     = f'-'

    #
    m_Lévy[2 * o + 1] = f'-'
    n_Lévy[2 * o + 1] = f'1'

    #
    m_Lévy[5] = f'5/12'
    m_Lévy[9] = f'5/12'
    n_Lévy[4] = f'5/12'
    n_Lévy[8] = f'5/12'

    # Create and open the text file
    with open(f'{simulation}/data/stability_analysis/Lévy_method.txt', 'w', encoding = 'utf8') as textfile:
        # Add a description of the contents
        textfile.write(fill(f'The sorted buckling load of a rectangular, specially orthotropic laminate with two simply-supported edges for'
                            f' various boundary conditions according to the Lévy method') )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the source and a timestamp
        textfile.write(fill(f'Source: .py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

        # Add an empty line
        textfile.write(f'\n\n')

        # Add the local stress at the top and bottom of each ply from the top to the bottom of the laminate
        textfile.write(tabulate(array( [boundary_conditions, applied_force, N_Lévy, m_Lévy, n_Lévy] ).T[N_Lévy.argsort()[::-1] ].tolist(),
                    headers = [f'Boundary conditions', f'Applied force', f'Buckling load', f'm [-]', f'n [-]'],
                    colalign = ('left', 'center', 'center', 'center', 'center'), floatfmt = ('.0f', '.0f', '.5e', '.0f', '.0f') ) )

    #
    return(N_Lévy, m_Lévy, n_Lévy)
