# Import modules
from datetime import datetime
from numpy    import einsum, ones, sum
from tabulate import tabulate
from textwrap import fill
from typing   import Dict, List, Tuple

# The function 'ABD_matrix'
def ABD_matrix(laminate : Dict[str, float or List[float] ], settings : Dict[str, str], stiffness : Dict[str, List[float] ],
               data : bool = False) -> Tuple[Dict[str, float or List[float] ], Dict[str, List[float] ] ]:
    """Computes the components of the ABD-matrix for a laminate.

    Returns the ``laminate`` dictionary appended with the height of the laminate (``h``) and an array of shape-(N + 1) containing the
    z-coordinate of each ply interface from the top to the bottom of the laminate (``z``), and the ``stiffness`` dictionary appended with three
    arrays of shape-(3, 3) containing the extensional stiffness (``A``-)matrix, the coupling stiffness (``B``-)matrix, and the bending
    stiffness (``D``-)matrix of the ABD-matrix for a laminate characterized by: the lay-up (``theta``), the thickness of the plies (``t``),
    and the transformed, reduced, stiffness matrix of each ply in the global coordinate system (``Q``). The computation is furthermore denoted
    by a moniker (``simulation``). Lastly, an optional boolean (``data``) can be defined which indicates if a text file (``ABD_matrix.txt``)
    containing the components of the ABD-matrix should be generated.

    Parameters
    ----------
    - ``laminate`` : Dict | Dictionary containing:
        - ``theta`` : numpy.ndarray, shape = (N) | The lay-up (in degrees) of the laminate (consisting of N plies).
        - ``t`` : float or numpy.ndarray, shape = (N) | The uniform thickness of each ply (float) or the thickness of each of the N plies
        separately (shape-(N) numpy.ndarray).
    - ``settings`` : Dict | Dictionary containing:
        - ``simulation`` : str | The moniker of the simulation.
    - ``stiffness`` : Dict | Dictionary containing:
        - ``Q`` : numpy.ndarray, shape = (3, 3, N) | The transformed, reduced, stiffness matrix in the global 1-2 coordinate system for each of
        the N plies.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the components of the ABD-matrix of the laminate should be
    generated, by default `False`.

    Returns
    -------
    - ``laminate`` : Dict | Dictionary appended with:
        - ``h`` : float | The height of the laminate.
        - ``z`` : numpy.ndarray, shape = (N + 1) | The z-coordinate of each ply interface from the top to the bottom of the laminate.
    - ``stiffness`` : Dict | Dictionary appended with:
        - ``A`` : numpy.ndarray, shape = (3, 3) | The extensional stiffness (A-)matrix of the laminate.
        - ``B`` : numpy.ndarray, shape = (3, 3) | The coupling stiffness (B-)matrix of the laminate.
        - ``D`` : numpy.ndarray, shape = (3, 3) | The bending stiffness (D-)matrix of the laminate.

    Output
    ------
    - ``ABD_matrix.txt`` : text file | A text file containing the components of the ABD-matrix of the laminate.

    Assumptions
    -----------
    - The properties of each ply are defined from the top to the bottom of the laminate.

    Version
    -------
    - v1.0 :
        - Initial version (06/08/2020) | M. Bilo

    References
    ----------
    [1]: Kassapoglou, Christos. 2010. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 1st ed.
    Chichester: John Wiley & Sons

    Verified with
    -------------
    [2]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.2 | NOTE: B₁₆ and B₂₆ are
    erroneous whereas B₆₁ and B₆₂ are correct as verified analytically.
    """

    # The lay-up of the laminate in degrees
    theta = laminate['theta']
    # The thickness of a/each ply
    t = laminate['t']

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3, N) containing the stiffness tensor of each of the N plies in the global 1-2 coordinate system
    Q = stiffness['Q']

    # The number of plies, N, comprising the laminate
    number_of_plies = len(theta)

    # An array of shape-(N) containing the thickness of each of the N plies
    t = t * ones(number_of_plies)

    # The height, h, of the laminate
    h = sum(t)

    # The ply interfaces and the corresponding z-coordinates are defined and demonstrated in figure 1 for N = 3 plies. In this figure can be
    # seen that the lay-up, and the N + 1 = 3 + 1 = 4 ply interfaces of the laminate, are defined from the top to the bottom of the laminate.
    # The elements of the corresponding arrays are indicated as well. The z-coordinate is based on the height (h) of the laminate and, as
    # shown, located at the midpoint between the top and bottom of the laminate (h / 2).
    #
    #                                   z
    #   0 +------------~ Top      ^     ^
    #     |   0. ply              |     |
    #   1 +------------~          |     |
    #     |   1. ply            h |   y +-------> x
    #   2 +------------~          |     ^
    #     |   2. ply              |     | h / 2
    #   3 +------------~ Bottom   v     v
    #
    # Figure 1: Schematic illustration of the ply interfaces of a laminate and the corresponding z-coordinate.

    # Create an array of shape-(N + 1) with a value of half the height of the laminate (h / 2)
    z = h / 2 * ones(number_of_plies + 1)
    # From the second ply interface on, successively compute the z-coordinate of each ply interface (the (i + 1)-th interface) by subtracting
    # the thickness of the current ply (the (i)-th ply) from the location of the previous ply interface (the (i)-th ply interface)
    for i in range(number_of_plies):
        # An array of shape-(N + 1) containing the z-coordinate of each of the N + 1 ply interfaces
        z[i + 1] = z[i] - t[i]

    # An array of shape-(3, 3) containing the extensional stiffness (A-)matrix (equation 3.41 from [1])
    A = einsum('ijk,k->ij', Q, t)

    # An array of shape-(3, 3) containing the coupling stiffness (B-)matrix (equation 3.50 from [1])
    B = einsum('ijk,k->ij', Q / 2, z[:-1]**2 - z[1:]**2)

    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix (equation 3.47 from [1])
    D = einsum('ijk,k->ij', Q / 3, z[:-1]**3 - z[1:]**3)

    # If a text file containing the components of the ABD-matrix of the laminate should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/ABD_matrix.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'Components of the ABD-matrix of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: ABD_matrix.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add the extensional stiffness (A-)matrix
            textfile.write(f'\n\n')
            textfile.write(f'A [N/m] =')
            textfile.write(f'\n')
            textfile.write(tabulate(A, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

            # Add the coupling stiffness (B-)matrix
            textfile.write(f'\n\n')
            textfile.write(f'B [N] =')
            textfile.write(f'\n')
            textfile.write(tabulate(B, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

            # Add the bending stiffness (D-)matrix
            textfile.write(f'\n\n')
            textfile.write(f'D [Nm] =')
            textfile.write(f'\n')
            textfile.write(tabulate(D, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

    # Append the laminate dictionary with the heigth of the laminate
    laminate['h'] = h
    # Append the laminate dictionary with the z-coordinate of each ply interface
    laminate['z'] = z

    # Append the stiffness dictionary with the extensional stiffness (A-)matrix
    stiffness['A'] = A
    # Append the stiffness dictionary with the coupling stiffness (B-)matrix
    stiffness['B'] = B
    # Append the stiffness dictionary with the bending stiffness (D-)matrix
    stiffness['D'] = D

    # End the function and return the appended laminate and stiffness dictionaries
    return(laminate, stiffness)
