# Import modules
from datetime     import datetime
from numpy.linalg import inv
from tabulate     import tabulate
from textwrap     import fill
from typing       import Dict, List

# The function 'inverse_ABD_matrix'
def ABD_matrix_inverse(settings : Dict[str, str], stiffness : Dict[str, List[float] ], data : bool = False) -> Dict[str, List[float] ]:
    """Computes the components of the inverse of the ABD-matrix for a laminate.

    Returns the ``stiffness`` dictionary appended with three arrays of shape-(3, 3) containing the extensional compliance matrix (``alpha``),
    the coupling compliance matrix (``beta``), and the bending compliance matrix (``delta``) of the inverse of the ABD-matrix for a laminate
    characterized by: the extensional stiffness (``A``-)matrix, the coupling stiffness (``B``-)matrix, and the bending stiffness (``D``-)matrix
    of the ABD-matrix. The computation is denoted by a moniker (``simulation``). Lastly, an optional boolean (``data``) can be defined which
    indicates if a text file (``inverse_ABD_matrix.txt``) containing the components of the inverse of the ABD-matrix should be generated.

    Parameters
    ----------
    - ``settings`` : Dict | Dictionary containing:
        - ``simulation`` : str | The moniker of the simulation.
    - ``stiffness`` : Dict | Dictionary containing:
        - ``A`` : numpy.ndarray, shape = (3, 3) | The extensional stiffness (A-)matrix of the laminate.
        - ``B`` : numpy.ndarray, shape = (3, 3) | The coupling stiffness (B-)matrix of the laminate.
        - ``D`` : numpy.ndarray, shape = (3, 3) | The bending stiffness (D-)matrix of the laminate.
    - ``data`` : bool, optional | Boolean indicating if a text file containing the components of the inverse of the ABD-matrix of the laminate
    should be generated, by default `False`.

    Returns
    -------
    - ``stiffness`` : Dict | Dictionary appended with:
        - ``alpha`` : numpy.ndarray, shape = (3, 3) | The extensional compliance matrix of the laminate.
        - ``beta`` : numpy.ndarray, shape = (3, 3) | The coupling compliance matrix of the laminate.
        - ``delta`` : numpy.ndarray, shape = (3, 3) | The bending compliance matrix of the laminate.

    Output
    ------
    - ``inverse_ABD_matrix.txt`` : text file | A text file containing the components of the inverse of the ABD-matrix of the laminate.

    Assumptions
    -----------

    References
    ----------
    [1]: Bernstein, Dennis. 2009. Matrix Mathematics: Theory, Facts and Formulas. 2nd ed. Woodstock: Princeton University Press

    [2]: Kassapoglou, Christos. 2010. Design and Analysis of Composite Structures: With Applications to Aerospace Structures. 1st ed.
    Chichester: John Wiley & Sons

    Verified with
    -------------
    [3]: Kaw, Autar. 2006. Mechanics of Composite Materials. 2nd ed. Boca Raton: Taylor & Francis Group, Example 4.4

    [*]: Numerically verified with: \n
         max(abs(concatenate( (concatenate( (alpha, beta), axis=1), concatenate( (beta, delta), axis=1) ) )
           - inv(concatenate( (concatenate( (A,     B),    axis=1), concatenate( (B,    D),     axis=1) ) ) ) ) )
    """

    # The moniker of the simulation
    simulation = settings['simulation']

    # An array of shape-(3, 3) containing the extensional stiffness (A-)matrix
    A = stiffness['A']
    # An array of shape-(3, 3) containing the coupling stiffness (B-)matrix
    B = stiffness['B']
    # An array of shape-(3, 3) containing the bending stiffness (D-)matrix
    D = stiffness['D']

    # An array of shape-(3, 3) containing the inverse of the extensional stiffness (A-)matrix
    A_inv = inv(A)

    # An array of shape-(3, 3) containing the bending compliance matrix (equation 2.8.16 from [1], and equation 3.54 from [2])
    delta = inv(D - B @ A_inv @ B)

    # An array of shape-(3, 3) containing the coupling compliance matrix (equation 2.8.16 from [1], and equations 3.53 and 3.54 from [2]).
    # NOTE: expression 3.53 from [2] is erroneous as verified with equation 2.8.16 from [1].
    beta = - A_inv @ B @ delta

    # An array of shape-(3, 3) containing the extensional compliance matrix (equation 2.8.16 from [1], and equations 3.52 and 3.53 from [2]).
    alpha = A_inv - beta @ B @ A_inv

    # If a text file containing the components of the inverse of the ABD-matrix of the laminate should be generated
    if data:
        # Create and open the text file
        with open(f'{simulation}/data/classical_laminated_plate_theory/ABD_matrix_inverse.txt', 'w', encoding = 'utf8') as textfile:
            # Add a description of the contents
            textfile.write(fill(f'The components of the inverse of the ABD-matrix of the laminate') )

            # Add an empty line
            textfile.write(f'\n\n')

            # Add the source and a timestamp
            textfile.write(fill(f'Source: ABD_matrix_inverse.py [v1.0] ({datetime.now().strftime("%H:%M:%S %d-%m-%Y")}).') )

            # Add the extensional compliance matrix
            textfile.write(f'\n\n')
            textfile.write(f'\u03B1 [m/N] =')
            textfile.write(f'\n')
            textfile.write(tabulate(alpha, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

            # Add the coupling compliance matrix
            textfile.write(f'\n\n')
            textfile.write(f'\u03B2 [1/N] =')
            textfile.write(f'\n')
            textfile.write(tabulate(beta, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

            # Add the bending compliance matrix
            textfile.write(f'\n\n')
            textfile.write(f'\u03B4 [1/Nm] =')
            textfile.write(f'\n')
            textfile.write(tabulate(delta, numalign = 'decimal', floatfmt = '.5e', tablefmt = 'plain') )

    # Append the stiffness dictionary with the extensional compliance matrix
    stiffness['alpha'] = alpha
    # Append the stiffness dictionary with the coupling compliance matrix
    stiffness['beta'] = beta
    # Append the stiffness dictionary with the bending compliance matrix
    stiffness['delta'] = delta

    # End the function and return the appended stiffness dictionary
    return(stiffness)
