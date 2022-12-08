# Import modules
from datetime import datetime
from numpy    import array, hstack, vstack
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List

# The function 'midplane_strain'
def midplane_strain(force : Dict[str, float], settings : Dict[str, str], stiffness : Dict[str, List[float] ],
                    data : bool = False) -> Dict[str, List[float] ]:
    """Computes the mid-plane strain and curvature due to the external and (fictitious) hygrothermal forces acting on the laminate.

    Returns a ``strain`` dictionary with an arrays of shape-(6) containing the mid-plane strains and curvatures due to the external and
    (fictitious) hygrothermal forces acting on the laminate. This laminate is characterized by: the extensional compliance matrix (``alpha``),
    the coupling compliance matrix (``beta``), the bending compliance matrix (``delta``), and the in-plane and flexural coefficients of thermal
    (``Alpha``) and moisture (``Beta``) expansion. The computation is denoted by a moniker (``simulation``). Lastly, an optional boolean
    (``data``) can be defined which indicates if a text file (``midplane_strain.txt``) containing the mid-plane strains and curvatures should
    be generated.

    Parameters
    ----------
    - ``force`` : Dict | Dictionary containing:
        - ``delta_C``: float | The moisture absorption or moisture content change.
        - ``delta_T``: float | The temperature difference/change.
        - ``N_x``: float | The distributed, external, normal force in longitudinal direction.
        - ``N_y``: float | The distributed, external, normal force in transverse direction.
        - ``N_xy``: float | The distributed, external, shear force in the longitudinal-transverse plane.
        - ``M_x``: float | The distributed, external, bending moment in the longitudinal direction.
        - ``M_y``: float | The distributed, external, bending moment in the transverse direction.
        - ``M_xy``: float | The distributed, external, twisting moment in the longitudinal-transverse plane.
    - ``settings`` : Dict | Dictionary containing:
        - ``simulation`` : str | The moniker of the simulation.
    - ``stiffness`` : Dict | Dictionary containing:
        - ``alpha`` : numpy.ndarray, shape = (3, 3) | The extensional compliance matrix of the laminate.
        - ``beta`` : numpy.ndarray, shape = (3, 3) | The coupling compliance matrix of the laminate.
        - ``delta`` : numpy.ndarray, shape = (3, 3) | The bending compliance matrix of the laminate.
        - ``Alpha`` : numpy.ndarray, shape = (6) | The in-plane and flexural coefficients of thermal expansion of the laminate.
        - ``Beta`` : numpy.ndarray, shape = (6) | The in-plane and flexural coefficients of moisture expansion of the laminate.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the mid-plane strains and curvatures in the laminate should be
    generated, by default `False`.

    Returns
    -------
    - ``strain`` : Dict | Dictionary containing:
        - ``epsilon_0`` : numpy.ndarray, shape = (6) | The mid-plane strains and curvatures in the laminate.

    Output
    ------
    - ``midplane_strain.txt`` : text file | A text file containing the mid-plane strains and curvatures in the laminate.

    Assumptions
    -----------

    References
    ----------
    [1]: Kassapoglou, Christos. 2010. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 1st ed.
    Chichester: John Wiley & Sons

    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group

    Verified with
    -------------
    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.3
    """

    # The moisture absorption or moisture content change
    delta_C = force['delta_C']
    # The temperature difference/change
    delta_T = force['delta_T']
    # The distributed, external, normal force in longitudinal direction
    N_x  = force['N_x']
    # The distributed, external, normal force in transverse direction
    N_y  = force['N_y']
    # The distributed, external, shear force in the longitudinal-transverse plane
    N_xy = force['N_xy']
    # The distributed, external, bending moment in the longitudinal direction
    M_x  = force['M_x']
    # The distributed, external, bending moment in the transverse direction
    M_y  = force['M_y']
    # The distributed, external, twisting moment in the longitudinal-transverse plane
    M_xy = force['M_xy']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the extensional compliance matrix
    alpha = stiffness['alpha']
    # An array of shape-(3, 3) containing the coupling compliance matrix
    beta = stiffness['beta']
    # An array of shape-(3, 3) containing the bending compliance matrix
    delta = stiffness['delta']
    # An array of shape-(6) containing the in-plane and flexural coefficients of thermal expansion
    Alpha = stiffness['Alpha']
    # An array of shape-(6) containing the in-plane and flexural coefficients of moisture expansion
    Beta = stiffness['Beta']

    # An array of shape-(6) containing the mechanical, external forces (equation 3.51 from [1])
    F_ext = array( [N_x, N_y, N_xy, M_x, M_y, M_xy] )

    # An array of shape-(6) containing the fictitious thermal loads (equations 4.64 and 4.65 from [2])
    F_the = delta_T * Alpha

    # An array of shape-(6) containing the fictitious moisture loads (equations 4.66 and 4.67 from [2])
    F_moi = delta_C * Beta

    # An array of shape-(6) containing the external and (fictitious) hygrothermal forces acting on the laminate (equation 4.68 from [2])
    F = F_ext + F_the + F_moi

    # An array of shape-(6, 6) containing the ABD-matrix (equation 3.51 from [1])
    abd = vstack( (hstack( (alpha, beta) ), hstack( (beta.T, delta) ) ) )

    # An array of shape-(6) containing the mid-plane strains and curvatures due to the external and (fictitious) hygrothermal forces acting on
    # the laminate (equation 3.51 from [1])
    epsilon_0 = abd @ F

    # If a text file containing the mid-plane strains and curvatures should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/midplane_strain.txt', 'w', encoding='utf8') as textfile:

            # Add a description of the contents
            textfile.write(fill(f'The mid-plane strains and curvatures due to external and (fictitious) hygrothermal loads') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: midplane_strain.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the mid-plane strains
            textfile.write(f'Mid-plane strains\n')
            textfile.write(tabulate([ [epsilon_0[0], epsilon_0[1], epsilon_0[2] ], [None, None, None] ],
                        headers = (f'\u03B5\u1D6A\u2080 [-]', f'\u03B5\u1D67\u2080 [-]', f'\u03B3\u1D6A\u1D67\u2080 [-]'),
                        stralign = 'left', numalign = 'decimal', floatfmt = '.5e') )

            # Add an empty line
            textfile.write(f'\n')

            # Add the curvatures
            textfile.write(f'Curvatures\n')
            textfile.write(tabulate([ [epsilon_0[3], epsilon_0[4], epsilon_0[5] ], [None, None, None] ],
                        headers = (f'\u03BA\u1D6A [1/m]', f'\u03BA\u1D67 [1/m]', f'\u03BA\u1D6A\u1D67 [1/m]'),
                        stralign = 'left', numalign = 'decimal', floatfmt = '.5e') )

    # Create the strain dictionary
    strain = {}
    # Append the strain dictionary with the mid-plane strains and curvatures
    strain['epsilon_0'] = epsilon_0

    # End the function and return the generated strain dictionary
    return(strain)
