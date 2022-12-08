# Import modules
from datetime import datetime
from numpy    import array, concatenate, cos, einsum, ones, pi, sin
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List

# The function 'thermal_coefficients'
def thermal_coefficients(laminate : Dict[str, float or List[float] ], material : Dict[str, float or List[float] ], settings : Dict[str, str],
                         stiffness : Dict[str, List[float] ], data: bool = False) -> Dict[str, List[float] ]:
    """Computes the in-plane and flexural coefficients of thermal expansion for a laminate.

    Returns the ``stiffness`` dictionary appended with a shape-6 array of in-plane and flexural coefficients of thermal expansion (corresponding
    to Nᵪ, Nᵧ, Nᵪᵧ, Mᵪ, Mᵧ, and Mᵪᵧ respectively) for a laminate characterized by: the lay-up (``theta``), the thickness of the plies (``t``), the
    z-coordinate of each ply interface (``z``), the longitudinal (``alpha_x``) and transverse (``alpha_y``) coefficient of thermal expansion, and the
    transformed, reduced, stiffness matrix of each ply in the global coordinate system (``Q``). The computation is furthermore denoted by a
    moniker (``simulation``). Lastly, an optional boolean (``data``) can be defined which indicates if a text file (``thermal_coefficients.txt``)
    containing the coefficients of thermal expansion should be generated.

    Parameters
    ----------
    - ``laminate`` : Dict | Dictionary containing:
        - ``theta`` : numpy.ndarray, shape = (N) | The lay-up (in degrees) of the laminate (consisting of N plies).
        - ``t`` : float or numpy.ndarray, shape = (N) | The uniform thickness of each ply (float) or the thickness of each of the N plies
        separately (shape-N numpy.ndarray).
        - ``z`` : numpy.ndarray, shape = (N + 1) | The z-coordinate of each of the N + 1 ply interfaces.
    - ``material`` : Dict | Dictionary containing:
        - ``alpha_x`` : float or numpy.ndarray, shape = (N) | The uniform longitudinal coefficient of thermal expansion of each ply (float)
        or the longitudinal coefficient of thermal expansion of each of the N plies separately (shape-N numpy.ndarray).
        - ``alpha_y`` : float or numpy.ndarray, shape = (N) | The uniform transverse coefficient of thermal expansion of each ply (float)
        or the transverse coefficient of thermal expansion of each of the N plies separately (shape-N numpy.ndarray).
    - ``settings`` : Dict | Dictionary containing:
        - ``simulation`` : str | The moniker of the simulation.
    - ``stiffness`` : Dict | Dictionary containing:
        - ``Q`` : numpy.ndarray, shape = (3, 3, N) | The transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each of
        the N plies.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the in-plane and flexural coefficients of thermal expansion should
    be generated, by default ``False``.

    Returns
    -------
    - ``stiffness`` : Dict | Dictionary appended with:
        - ``Alpha`` : numpy.ndarray, shape = (6) | The in-plane and flexural coefficients of thermal expansion of the laminate.

    Output
    ------
    - ``thermal_coefficients.txt`` : text file | A text file containing the in-plane and flexural coefficients of thermal expansion of the
    laminate.

    Assumptions
    -----------
    - The laminate with width a, length b and height h is assumed to be thin (a, b > 10h).
    - The deformation is partly assumed to be geometrically linear:
        - The displacements in the x, y, and z-direction (u, v, and w respectively) are assumed to be small (u, v, w << h).
        - The in-plane strains (ϵᵪ, ϵᵧ, and ϵᵪᵧ) are small.
        - Non-linear terms in the equations of motion involving products of stresses and plate slopes are retained to include in=plane force effects.
    - Plane stress is assumed:
        - The transverse shear strains (ϵ and ϵ) are zero/negligible.
    - The classical assumptions of Kirchhoff hold:
        - asdf
    - Perfect bonding between layers is assumed:
        -
        -
        -

    Version
    -------
    - v1.0 :
        - Initial version (04/08/2020) | M. Bilo

    References
    ----------
    [1]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    Verified with
    -------------
    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Examples 2.20 & 4.5 | NOTE: The global
    stresses given in table 4.6 for example 4.5 are erroneous as verified analytically.
    """

    # The lay-up of the laminate in degrees
    theta = laminate['theta']
    # The thickness of a/each ply
    t = laminate['t']
    # The z-coordinate of each ply interface
    z = laminate['z']

    # The longitudinal coefficient of thermal expansion of a/each ply
    alpha_x = material['alpha_x']
    # The transverse coefficient of thermal expansion of a/each ply
    alpha_y = material['alpha_y']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3, N) containing the stiffness tensor of each of the N plies in the global 1-2 coordinate system
    Q = stiffness['Q']

    # The number of plies, N, comprising the laminate
    number_of_plies = len(theta)

    # An array of shape-(N) containing the thickness of each of the N plies
    t = t * ones(number_of_plies)

    # An array of shape-(N) containing the longitudinal coefficient of moisture of each of the N plies
    alpha_x = alpha_x * ones(number_of_plies)

    # An array of shape-(N) containing the transverse coefficient of moisture of each of the N plies
    alpha_y = alpha_y * ones(number_of_plies)

    # Two arrays of shape-(N) containing recurring components of the standard, tensor transformation matrix for each of the N plies (equations
    # 2.97a and 2.97b from [1])
    c = cos(2 * pi * theta / 360)
    s = sin(2 * pi * theta / 360)

    # An array of shape-(3, 2, N) containing the inverse of the reduced transformation matrix for each of the N plies (equations 2.95 and 2.181
    # from [1])
    T = array( [ [    c * c,       s * s],
                 [    s * s,       c * c],
                 [2 * c * s, - 2 * c * s] ] )

    # An array of shape-(3, N) containing the 3 coefficients of thermal expansion (alpha_1, alpha_2, and alpha_12 respectively) for each of the
    # N plies (equations 2.95 and 2.181 from [1])
    alpha = einsum('ijk,jk->ik', T, array([alpha_x, alpha_y] ) )

    # An array of shape-(6) containing the in-plane and flexural coefficients of thermal expansion of the laminate which correspond to the
    # fictitious thermal loads Nᵪ, Nᵧ, Nᵪᵧ, Mᵪ, Mᵧ, and Mᵪᵧ respectively (equations 4.64 and 4.65 from [1])
    Alpha = concatenate( (einsum('ik,k->i', einsum('jk,ijk->ik', alpha, Q), t),
                          einsum('ik,k->i', einsum('jk,ijk->ik', alpha, Q), (z[:-1]**2 - z[1:]**2) / 2) ) )

    # If a text file containing the coefficients of thermal expansion of the laminate should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/thermal_coefficients.txt', 'w', encoding='utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'In-plane and flexural coefficients of thermal expansion of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: thermal_coefficients.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d/%m/%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the in-plane coefficients of thermal expansion of the laminate
            textfile.write(f'In-plane coefficients of thermal expansion \n')
            textfile.write(tabulate( [ [Alpha[0], Alpha[1], Alpha[2] ], [None, None, None] ],
                                    headers=(f'\u03B1\u1D6A\u2098 [N/(K\u00B7m)]', f'\u03B1\u1D67\u2098 [N/(K\u00B7m)]',
                                             f'\u03B1\u1D6A\u1D67\u2098 [N/(K\u00B7m)]'),
                                    stralign='center', numalign='center', floatfmt='.5e') )

            # Add an empty line
            textfile.write(f'\n')

            # Add the flexural coefficients of thermal expansion of the laminate
            textfile.write(f'Flexural coefficients of thermal expansion \n')
            textfile.write(tabulate( [ [Alpha[3], Alpha[4], Alpha[5] ], [None, None, None] ],
                                    headers=(f'\u03B1\u1D6A\u2095 [N/K]', f'\u03B1\u1D67\u2095 [N/K]',
                                             f'\u03B1\u1D6A\u1D67\u2095 [N/K]'),
                                    stralign='center', numalign='center', floatfmt='.5e') )

    # Append the stiffness dictionary with the in-plane and flexural coefficients of thermal expansion of the laminate
    stiffness['Alpha'] = Alpha

    # End the function and return the appended stiffness dictionary
    return(stiffness)
