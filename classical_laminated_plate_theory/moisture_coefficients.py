# Import modules
from datetime  import datetime
from jax.numpy import array, concatenate, cos, deg2rad, einsum, ones, sin
from tabulate  import tabulate
from textwrap  import fill

# The function 'moisture_coefficients'
def moisture_coefficients(laminate : dict[list[float], list[float], list[float] ], material : dict[list[float],list[float] ],
                         settings : dict[str], stiffness : dict[list[float] ], data: bool = False) -> dict[list[float] ]:
    """Computes the in-plane and flexural coefficients of moisture expansion for a laminate.

    Returns the `stiffness` dictionary appended with a shape-6 array of in-plane and flexural coefficients of moisture expansion (corresponding
    to Nᵪ, Nᵧ, Nᵪᵧ, Mᵪ, Mᵧ, and Mᵪᵧ respectively) for a laminate characterized by: the lay-up (`theta`), the thickness of the plies (`t`), the
    z-coordinate of each ply interface (`z`), the longitudinal (`beta_x`) and transverse (`beta_y`) coefficient of moisture expansion, and the
    transformed, reduced, stiffness matrix of each ply in the global coordinate system (`Q`). The computation is furthermore denoted by a
    moniker (`simulation`). Lastly, an optional boolean (`data`) can be defined which indicates if a text file (`moisture_coefficients.txt`)
    containing the coefficients of moisture expansion should be generated.

    Parameters
    ----------
    - `laminate` : Dict | Dictionary containing:
        - `theta` : jax.numpy.ndarray, shape = (N) | The lay-up (in degrees) of the laminate (consisting of N plies).
        - `t` : float or jax.numpy.ndarray, shape = (N) | The uniform thickness of each ply (float) or the thickness of each of the N plies
        separately (shape-N jax.numpy.ndarray).
        - `z` : jax.numpy.ndarray, shape = (N + 1) | The z-coordinate of each of the N + 1 ply interfaces.
    - `material` : Dict | Dictionary containing:
        - `beta_x` : float or jax.numpy.ndarray, shape = (N) | The uniform longitudinal coefficient of moisture expansion of each ply (float)
        or the longitudinal coefficient of moisture expansion of each of the N plies separately (shape-N jax.numpy.ndarray).
        - `beta_y` : float or jax.numpy.ndarray, shape = (N) | The uniform transverse coefficient of moisture expansion of each ply (float)
        or the transverse coefficient of moisture expansion of each of the N plies separately (shape-N jax.numpy.ndarray).
    - `settings` : Dict | Dictionary containing:
        - `simulation` : str | The moniker of the simulation.
    - `stiffness` : Dict | Dictionary containing:
        - `Q` : jax.numpy.ndarray, shape = (3, 3, N) | The transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each
        of the N plies.
    - `data` : bool, optional | Boolean indicating if a text file containing the in-plane and flexural coefficients of moisture expansion should
    be generated, by default `False`.

    Returns
    -------
    - `stiffness` : Dict | Dictionary appended with:
        - `Beta` : jax.numpy.ndarray, shape = 6 | The in-plane and flexural coefficients of moisture expansion of the laminate.

    Output
    ------
    - `moisture_coefficients.txt` : text file | A text file containing the in-plane and flexural coefficients of moisture expansion of the
    laminate.

    Assumptions
    -----------

    Version
    -------
    - v1.0 :
        - Initial version (04/08/2020) | M. Bilo
    - v1.1 :
        - Updated documentation (14/11/2024) | M. Bilo

    References
    ----------
    [1]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    Verified with
    -------------
    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 2.20
    """

    # Version of the script
    version = f'v1.1'

    # The lay-up of the laminate in degrees
    theta = laminate['theta']
    # The thickness of a / each ply
    t = laminate['t']
    # The z-coordinate of each ply interface
    z = laminate['z']

    # The longitudinal coefficient of moisture expansion of a / each ply
    beta_x = material['beta_x']
    # The transverse coefficient of moisture expansion of a / each ply
    beta_y = material['beta_y']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3, N) containing the stiffness tensor of each of the N plies in the global 1-2 coordinate system
    Q = stiffness['Q']

    # The number of plies, N, comprising the laminate
    number_of_plies = len(theta)

    # An array of shape-(N) containing the thickness of each of the N plies
    t = (isinstance(t, float) ) * t * ones(number_of_plies) + (not isinstance(t, float) ) * t

    # An array of shape-(N) containing the longitudinal coefficient of moisture of each of the N plies
    beta_x = beta_x * ones(number_of_plies)

    # An array of shape-(N) containing the transverse coefficient of moisture of each of the N plies
    beta_y = beta_y * ones(number_of_plies)

    # Two arrays of shape-(N) containing recurring components of the standard, tensor transformation matrix for each of the N plies (equations
    # 2.97a and 2.97b from [1])
    #
    # c = cos(θ), and
    #
    # s = sin(θ)
    #
    c = cos(deg2rad(theta) )
    s = sin(deg2rad(theta) )

    # An array of shape-(3, 2, N) containing the inverse of the reduced transformation matrix for each of the N plies (equations 2.95 and 2.181
    # from [1])
    #
    #
    #
    #
    #
    T = array( [ [    c * c,       s * s],
                 [    s * s,       c * c],
                 [2 * c * s, - 2 * c * s] ] )

    # An array of shape-(3, N) containing the 3 coefficients of moisture expansion (beta_1, beta_2, and beta_12 respectively) for each of the
    # N plies (equations 2.95 and 2.182 from [1])
    #
    # |
    # |
    # |
    #
    beta = einsum('ijk,jk->ik', T, array([beta_x, beta_y] ) )

    # An array of shape-(6) containing the in-plane and flexural coefficients of moisture expansion of the laminate which correspond to the
    # fictitious moisture loads Nᵪ, Nᵧ, Nᵪᵧ, Mᵪ, Mᵧ, and Mᵪᵧ respectively (equations 4.66 and 4.67 from [1])
    #
    #
    #
    #
    #
    #
    #
    Beta = concatenate( (einsum('ik,k->i', einsum('jk,ijk->ik', beta, Q), t),
                         einsum('ik,k->i', einsum('jk,ijk->ik', beta, Q), (z[:-1]**2 - z[1:]**2) / 2) ) )

    # If a text file containing the coefficients of moisture expansion of the laminate should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/moisture_coefficients.txt', 'w', encoding='utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'In-plane and flexural coefficients of moisture expansion of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: moisture_coefficients.py [{version}] ({datetime.now().strftime("%H:%M:%S %d/%m/%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the in-plane coefficients of moisture expansion of the laminate
            textfile.write(f'In-plane coefficients of moisture expansion \n')
            textfile.write(tabulate( [ [Beta[0], Beta[1], Beta[2] ], [None, None, None] ],
                                    headers=(f'\u03B2\u1D6A\u2098 [N/m]', f'\u03B2\u1D67\u2098 [N/m]',
                                             f'\u03B2\u1D6A\u1D67\u2098 [N/m]'),
                                    stralign='center', numalign='center', floatfmt='.5e') )

            # Add an empty line
            textfile.write(f'\n')

            # Add the flexural coefficients of moisture expansion of the laminate
            textfile.write(f'Flexural coefficients of moisture expansion \n')
            textfile.write(tabulate( [ [Beta[3], Beta[4], Beta[5] ], [None, None, None] ],
                                    headers=(f'\u03B2\u1D6A\u2095 [N]', f'\u03B2\u1D67\u2095 [N]',
                                             f'\u03B2\u1D6A\u1D67\u2095 [N]'),
                                    stralign='center', numalign='center', floatfmt='.5e') )

    # Append the stiffness dictionary with the in-plane and flexural coefficients of moisture expansion of the laminate
    stiffness['Beta'] = Beta

    # End the function and return the appended stiffness dictionary
    return(stiffness)
